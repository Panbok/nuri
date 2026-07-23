#include "nuri/tools/autotest/autotest_timeline.h"

#include <algorithm>
#include <cmath>
#include <map>

namespace nuri::tools::autotest {
namespace {

[[nodiscard]] float smoothstep(float t) { return t * t * (3.0f - 2.0f * t); }

[[nodiscard]] AutotestCameraConfig
cameraFromKeyframe(const AutotestCase &testCase,
                   const AutotestCameraKeyframe &keyframe) {
  AutotestCameraConfig camera = testCase.camera;
  camera.position = keyframe.position;
  if (keyframe.hasTarget) {
    camera.target = keyframe.target;
    camera.hasTarget = true;
    const glm::vec3 delta = keyframe.target - keyframe.position;
    if (glm::length(delta) > 1.0e-6f) {
      camera.direction = glm::normalize(delta);
    }
  }
  return camera;
}

[[nodiscard]] bool frameInPath(const AutotestCameraPath &path, uint32_t frame) {
  return frame >= path.startFrame && frame <= path.endFrame &&
         !path.keyframes.empty();
}

[[nodiscard]] Result<AutotestCameraConfig, std::string>
evaluatePath(const AutotestCase &testCase, const AutotestCameraPath &path,
             uint32_t frame) {
  if (path.keyframes.empty()) {
    return Result<AutotestCameraConfig, std::string>::makeError(
        "camera path '" + path.id + "' has no keyframes");
  }
  const auto upper =
      std::lower_bound(path.keyframes.begin(), path.keyframes.end(), frame,
                       [](const AutotestCameraKeyframe &keyframe,
                          uint32_t value) { return keyframe.frame < value; });
  if (upper == path.keyframes.begin()) {
    return Result<AutotestCameraConfig, std::string>::makeResult(
        cameraFromKeyframe(testCase, *upper));
  }
  if (upper == path.keyframes.end()) {
    return Result<AutotestCameraConfig, std::string>::makeResult(
        cameraFromKeyframe(testCase, path.keyframes.back()));
  }
  const AutotestCameraKeyframe &rhs = *upper;
  const AutotestCameraKeyframe &lhs = *(upper - 1);
  const uint32_t span = rhs.frame - lhs.frame;
  float t = span == 0u ? 0.0f
                       : static_cast<float>(frame - lhs.frame) /
                             static_cast<float>(span);
  if (path.interpolation == "smoothstep") {
    t = smoothstep(t);
  } else if (path.interpolation != "linear") {
    return Result<AutotestCameraConfig, std::string>::makeError(
        "camera path '" + path.id + "' has unsupported interpolation '" +
        path.interpolation + "'");
  }

  AutotestCameraConfig camera = testCase.camera;
  camera.position = glm::mix(lhs.position, rhs.position, t);
  const bool hasTarget = lhs.hasTarget && rhs.hasTarget;
  if (hasTarget) {
    camera.target = glm::mix(lhs.target, rhs.target, t);
    camera.hasTarget = true;
    const glm::vec3 delta = camera.target - camera.position;
    if (glm::length(delta) > 1.0e-6f) {
      camera.direction = glm::normalize(delta);
    }
  }
  return Result<AutotestCameraConfig, std::string>::makeResult(camera);
}

} // namespace

Result<AutotestCameraConfig, std::string>
evaluateAutotestCameraAtFrame(const AutotestCase &testCase, uint32_t frame) {
  AutotestCameraConfig camera = testCase.camera;
  for (const AutotestTimelineEvent &event : testCase.timeline.events) {
    if (event.frame <= frame && event.type == "setCamera") {
      camera = event.camera;
    }
  }
  for (const AutotestCameraPath &path : testCase.timeline.cameraPaths) {
    if (!frameInPath(path, frame)) {
      continue;
    }
    auto evaluated = evaluatePath(testCase, path, frame);
    if (evaluated.hasError()) {
      return evaluated;
    }
    camera = evaluated.value();
  }
  return Result<AutotestCameraConfig, std::string>::makeResult(camera);
}

Result<std::vector<AutotestFramePlan>, std::string>
compileAutotestTimeline(const AutotestCase &testCase) {
  std::map<uint32_t, AutotestFramePlan> frames;
  RenderSettings currentSettings = testCase.settings;
  for (uint32_t frame = 0u; frame <= testCase.endFrame; ++frame) {
    for (const AutotestTimelineEvent &event : testCase.timeline.events) {
      if (event.frame <= frame && event.type == "setSettings" &&
          event.hasSettings) {
        currentSettings = event.settings;
      }
    }
    auto camera = evaluateAutotestCameraAtFrame(testCase, frame);
    if (camera.hasError()) {
      return Result<std::vector<AutotestFramePlan>, std::string>::makeError(
          camera.error());
    }
    frames.emplace(frame, AutotestFramePlan{.frame = frame,
                                            .camera = camera.value(),
                                            .settings = currentSettings});
  }

  for (const AutotestTimelineEvent &event : testCase.timeline.events) {
    AutotestFramePlan &plan = frames[event.frame];
    if (event.type == "resetTemporalHistory") {
      plan.resetTemporalHistory = true;
      plan.resetReason = event.eventReason;
    } else if (event.type == "setCamera") {
      plan.camera = event.camera;
      if (!event.preserveHistory) {
        plan.cameraCut = true;
      }
    } else if (event.type == "setSettings") {
      if (!event.hasSettings) {
        return Result<std::vector<AutotestFramePlan>, std::string>::makeError(
            "setSettings event requires settings");
      }
      plan.settings = event.settings;
    } else if (event.type == "setDirectionalLightIntensity" ||
               event.type == "setLocalLightIntensity" ||
               event.type == "setNodeTranslation" ||
               event.type == "setDDGIVolumeProbeCounts") {
      plan.sceneEvents.push_back(&event);
    } else {
      return Result<std::vector<AutotestFramePlan>, std::string>::makeError(
          "unsupported timeline event type '" + event.type + "'");
    }
  }

  for (const AutotestCheckpoint &checkpoint : testCase.checkpoints) {
    frames[checkpoint.frame].checkpoints.push_back(&checkpoint);
  }

  std::vector<AutotestFramePlan> out;
  out.reserve(frames.size() + kAutotestReadoutDrainFrameLimit);
  for (auto &[frame, plan] : frames) {
    (void)frame;
    out.push_back(plan);
  }
  const bool hasReadouts =
      std::any_of(testCase.checkpoints.begin(), testCase.checkpoints.end(),
                  [](const AutotestCheckpoint &checkpoint) {
                    return !checkpoint.readouts.empty();
                  });
  if (hasReadouts && !out.empty()) {
    const AutotestFramePlan finalPlanned = out.back();
    for (uint32_t i = 1u; i <= kAutotestReadoutDrainFrameLimit; ++i) {
      AutotestFramePlan drain = finalPlanned;
      drain.frame = finalPlanned.frame + i;
      drain.resetTemporalHistory = false;
      drain.cameraCut = false;
      drain.drainOnly = true;
      drain.resetReason.clear();
      drain.checkpoints.clear();
      out.push_back(std::move(drain));
    }
  }
  return Result<std::vector<AutotestFramePlan>, std::string>::makeResult(
      std::move(out));
}

} // namespace nuri::tools::autotest

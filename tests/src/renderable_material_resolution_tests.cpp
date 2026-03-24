#include "tests_pch.h"

#include "nuri/gfx/layers/renderable_material_resolution.h"

namespace {

TEST(RenderableMaterialResolutionTests,
     PrefersOverrideThenModelThenFallbackMaterial) {
  nuri::Renderable renderable{};
  renderable.material = nuri::makeMaterialRef(10u, 1u);

  nuri::ModelRecord modelRecord;
  modelRecord.sourceMaterialToRuntime.push_back(nuri::makeMaterialRef(20u, 1u));

  EXPECT_EQ(nuri::resolveRenderableMaterial(renderable, modelRecord, 0u).value,
            nuri::makeMaterialRef(20u, 1u).value);

  renderable.materialOverride = nuri::makeMaterialRef(30u, 1u);
  EXPECT_EQ(nuri::resolveRenderableMaterial(renderable, modelRecord, 0u).value,
            nuri::makeMaterialRef(30u, 1u).value);

  renderable.materialOverride = nuri::kInvalidMaterialRef;
  modelRecord.sourceMaterialToRuntime[0] = nuri::kInvalidMaterialRef;
  EXPECT_EQ(nuri::resolveRenderableMaterial(renderable, modelRecord, 0u).value,
            renderable.material.value);
}

TEST(RenderableMaterialResolutionTests,
     ReturnsInvalidMaterialWhenSourceMaterialMapIsEmpty) {
  nuri::Renderable renderable{};
  nuri::ModelRecord modelRecord;

  const nuri::MaterialRef resolved =
      nuri::resolveRenderableMaterial(renderable, modelRecord, 0u);
  EXPECT_EQ(resolved.value, nuri::kInvalidMaterialRef.value);
}

TEST(RenderableMaterialResolutionTests,
     ReturnsInvalidMaterialWhenSourceMaterialIndexIsOutOfBounds) {
  nuri::Renderable renderable{};
  nuri::ModelRecord modelRecord;
  modelRecord.sourceMaterialToRuntime.push_back(nuri::makeMaterialRef(20u, 1u));
  modelRecord.sourceMaterialToRuntime.push_back(nuri::makeMaterialRef(21u, 1u));

  const nuri::MaterialRef resolved =
      nuri::resolveRenderableMaterial(renderable, modelRecord, 2u);
  EXPECT_EQ(resolved.value, nuri::kInvalidMaterialRef.value);
}

} // namespace

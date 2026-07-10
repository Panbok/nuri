#include "nuri/tools/core/process.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <system_error>
#include <thread>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <csignal>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace nuri::tools::core {
namespace {

using Clock = std::chrono::steady_clock;

void setElapsed(ProcessResult &result, const Clock::time_point started) {
  result.elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      Clock::now() - started);
}

#if defined(_WIN32)

class UniqueHandle {
public:
  UniqueHandle() = default;
  explicit UniqueHandle(HANDLE handle) : handle_(handle) {}
  UniqueHandle(const UniqueHandle &) = delete;
  UniqueHandle &operator=(const UniqueHandle &) = delete;
  UniqueHandle(UniqueHandle &&other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)) {}
  UniqueHandle &operator=(UniqueHandle &&other) noexcept {
    if (this != &other) {
      reset(std::exchange(other.handle_, nullptr));
    }
    return *this;
  }
  ~UniqueHandle() { reset(); }

  [[nodiscard]] HANDLE get() const noexcept { return handle_; }
  [[nodiscard]] HANDLE release() noexcept {
    return std::exchange(handle_, nullptr);
  }
  void reset(HANDLE handle = nullptr) noexcept {
    if (handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE) {
      CloseHandle(handle_);
    }
    handle_ = handle;
  }

private:
  HANDLE handle_ = nullptr;
};

[[nodiscard]] std::string windowsErrorMessage(const DWORD error) {
  wchar_t *wide = nullptr;
  const DWORD count = FormatMessageW(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, error, 0u, reinterpret_cast<wchar_t *>(&wide), 0u, nullptr);
  if (count == 0u || wide == nullptr) {
    return "Windows error " + std::to_string(error);
  }

  const int utf8Count = WideCharToMultiByte(
      CP_UTF8, 0u, wide, static_cast<int>(count), nullptr, 0, nullptr, nullptr);
  std::string message;
  if (utf8Count > 0) {
    message.resize(static_cast<size_t>(utf8Count));
    WideCharToMultiByte(CP_UTF8, 0u, wide, static_cast<int>(count),
                        message.data(), utf8Count, nullptr, nullptr);
    while (!message.empty() &&
           (message.back() == '\r' || message.back() == '\n')) {
      message.pop_back();
    }
  }
  LocalFree(wide);
  return message.empty() ? "Windows error " + std::to_string(error) : message;
}

[[nodiscard]] bool utf8ToWide(const std::string &text, std::wstring &wide,
                              std::string &errorMessage) {
  if (text.empty()) {
    wide.clear();
    return true;
  }
  const int count =
      MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
                          static_cast<int>(text.size()), nullptr, 0);
  if (count <= 0) {
    errorMessage =
        "argument is not valid UTF-8: " + windowsErrorMessage(GetLastError());
    return false;
  }
  wide.resize(static_cast<size_t>(count));
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
                      static_cast<int>(text.size()), wide.data(), count);
  return true;
}

[[nodiscard]] std::wstring quoteWindowsArgument(const std::wstring &argument) {
  if (!argument.empty() &&
      argument.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
    return argument;
  }

  std::wstring quoted(1u, L'\"');
  size_t backslashes = 0u;
  for (const wchar_t character : argument) {
    if (character == L'\\') {
      ++backslashes;
      continue;
    }
    if (character == L'\"') {
      quoted.append(backslashes * 2u + 1u, L'\\');
      quoted.push_back(L'\"');
      backslashes = 0u;
      continue;
    }
    quoted.append(backslashes, L'\\');
    backslashes = 0u;
    quoted.push_back(character);
  }
  quoted.append(backslashes * 2u, L'\\');
  quoted.push_back(L'\"');
  return quoted;
}

[[nodiscard]] bool createCapturePipe(UniqueHandle &readHandle,
                                     UniqueHandle &writeHandle,
                                     std::string &errorMessage) {
  SECURITY_ATTRIBUTES attributes{};
  attributes.nLength = sizeof(attributes);
  attributes.bInheritHandle = TRUE;
  HANDLE rawRead = nullptr;
  HANDLE rawWrite = nullptr;
  if (!CreatePipe(&rawRead, &rawWrite, &attributes, 0u)) {
    errorMessage = "CreatePipe failed: " + windowsErrorMessage(GetLastError());
    return false;
  }
  readHandle.reset(rawRead);
  writeHandle.reset(rawWrite);
  if (!SetHandleInformation(readHandle.get(), HANDLE_FLAG_INHERIT, 0u)) {
    errorMessage =
        "SetHandleInformation failed: " + windowsErrorMessage(GetLastError());
    return false;
  }
  return true;
}

void readPipe(HANDLE pipe, std::string &output) {
  UniqueHandle owned(pipe);
  std::array<char, 16u * 1024u> buffer{};
  for (;;) {
    DWORD bytesRead = 0u;
    if (!ReadFile(owned.get(), buffer.data(), static_cast<DWORD>(buffer.size()),
                  &bytesRead, nullptr)) {
      break;
    }
    if (bytesRead == 0u) {
      break;
    }
    output.append(buffer.data(), static_cast<size_t>(bytesRead));
  }
}

[[nodiscard]] DWORD waitDuration(const std::chrono::milliseconds timeout) {
  if (timeout == (std::chrono::milliseconds::max)()) {
    return INFINITE;
  }
  if (timeout.count() <= 0) {
    return 0u;
  }
  return static_cast<DWORD>(
      std::min<int64_t>(timeout.count(), static_cast<int64_t>(INFINITE) - 1));
}

ProcessResult runProcessPlatform(const ProcessCommand &command,
                                 const ProcessOptions &options,
                                 const Clock::time_point started) {
  ProcessResult result{};
  if (command.executable.empty()) {
    result.errorMessage = "process executable is empty";
    setElapsed(result, started);
    return result;
  }

  std::wstring commandLine = quoteWindowsArgument(command.executable.native());
  for (const std::string &argument : command.arguments) {
    std::wstring wide;
    if (!utf8ToWide(argument, wide, result.errorMessage)) {
      setElapsed(result, started);
      return result;
    }
    commandLine.push_back(L' ');
    commandLine += quoteWindowsArgument(wide);
  }

  UniqueHandle stdoutRead;
  UniqueHandle stdoutWrite;
  UniqueHandle stderrRead;
  UniqueHandle stderrWrite;
  if (!createCapturePipe(stdoutRead, stdoutWrite, result.errorMessage) ||
      !createCapturePipe(stderrRead, stderrWrite, result.errorMessage)) {
    setElapsed(result, started);
    return result;
  }

  SECURITY_ATTRIBUTES attributes{};
  attributes.nLength = sizeof(attributes);
  attributes.bInheritHandle = TRUE;
  UniqueHandle stdinHandle(
      CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                  &attributes, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
  if (stdinHandle.get() == INVALID_HANDLE_VALUE) {
    result.errorMessage = "opening NUL for child stdin failed: " +
                          windowsErrorMessage(GetLastError());
    setElapsed(result, started);
    return result;
  }

  SIZE_T attributeBytes = 0u;
  InitializeProcThreadAttributeList(nullptr, 1u, 0u, &attributeBytes);
  std::vector<std::byte> attributeStorage(attributeBytes);
  auto *attributeList =
      reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributeStorage.data());
  if (!InitializeProcThreadAttributeList(attributeList, 1u, 0u,
                                         &attributeBytes)) {
    result.errorMessage = "InitializeProcThreadAttributeList failed: " +
                          windowsErrorMessage(GetLastError());
    setElapsed(result, started);
    return result;
  }
  struct AttributeListGuard {
    LPPROC_THREAD_ATTRIBUTE_LIST list;
    ~AttributeListGuard() { DeleteProcThreadAttributeList(list); }
  } attributeGuard{attributeList};

  std::array<HANDLE, 3u> inheritedHandles{stdinHandle.get(), stdoutWrite.get(),
                                          stderrWrite.get()};
  if (!UpdateProcThreadAttribute(attributeList, 0u,
                                 PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                 inheritedHandles.data(),
                                 sizeof(inheritedHandles), nullptr, nullptr)) {
    result.errorMessage = "UpdateProcThreadAttribute failed: " +
                          windowsErrorMessage(GetLastError());
    setElapsed(result, started);
    return result;
  }

  STARTUPINFOEXW startup{};
  startup.StartupInfo.cb = sizeof(startup);
  startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  startup.StartupInfo.hStdInput = stdinHandle.get();
  startup.StartupInfo.hStdOutput = stdoutWrite.get();
  startup.StartupInfo.hStdError = stderrWrite.get();
  startup.lpAttributeList = attributeList;

  PROCESS_INFORMATION processInfo{};
  std::wstring workingDirectory;
  const wchar_t *workingDirectoryPointer = nullptr;
  if (options.workingDirectory.has_value()) {
    workingDirectory = options.workingDirectory->native();
    workingDirectoryPointer = workingDirectory.c_str();
  }

  std::vector<wchar_t> mutableCommandLine(commandLine.begin(),
                                          commandLine.end());
  mutableCommandLine.push_back(L'\0');
  const DWORD creationFlags = EXTENDED_STARTUPINFO_PRESENT | CREATE_SUSPENDED |
                              CREATE_UNICODE_ENVIRONMENT;
  if (!CreateProcessW(command.executable.c_str(), mutableCommandLine.data(),
                      nullptr, nullptr, TRUE, creationFlags, nullptr,
                      workingDirectoryPointer, &startup.StartupInfo,
                      &processInfo)) {
    result.errorMessage =
        "CreateProcess failed: " + windowsErrorMessage(GetLastError());
    setElapsed(result, started);
    return result;
  }

  UniqueHandle process(processInfo.hProcess);
  UniqueHandle thread(processInfo.hThread);
  UniqueHandle job(CreateJobObjectW(nullptr, nullptr));
  if (job.get() != nullptr) {
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags =
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!SetInformationJobObject(job.get(), JobObjectExtendedLimitInformation,
                                 &limits, sizeof(limits)) ||
        !AssignProcessToJobObject(job.get(), process.get())) {
      job.reset();
    }
  }

  stdoutWrite.reset();
  stderrWrite.reset();
  stdinHandle.reset();

  std::thread stdoutThread;
  std::thread stderrThread;
  try {
    stdoutThread = std::thread(readPipe, stdoutRead.release(),
                               std::ref(result.standardOutput));
    stderrThread = std::thread(readPipe, stderrRead.release(),
                               std::ref(result.standardError));
  } catch (const std::system_error &error) {
    TerminateProcess(process.get(), 1u);
    WaitForSingleObject(process.get(), INFINITE);
    job.reset();
    if (stdoutThread.joinable()) {
      stdoutThread.join();
    }
    if (stderrThread.joinable()) {
      stderrThread.join();
    }
    result.status = ProcessStatus::InternalError;
    result.errorMessage =
        "starting output capture failed: " + std::string(error.what());
    setElapsed(result, started);
    return result;
  }

  if (ResumeThread(thread.get()) == (std::numeric_limits<DWORD>::max)()) {
    TerminateProcess(process.get(), 1u);
    WaitForSingleObject(process.get(), INFINITE);
    job.reset();
    stdoutThread.join();
    stderrThread.join();
    result.status = ProcessStatus::InternalError;
    result.errorMessage =
        "ResumeThread failed: " + windowsErrorMessage(GetLastError());
    setElapsed(result, started);
    return result;
  }

  const DWORD waitResult =
      WaitForSingleObject(process.get(), waitDuration(options.timeout));
  if (waitResult == WAIT_TIMEOUT) {
    if (job.get() != nullptr) {
      TerminateJobObject(job.get(), 1u);
    } else {
      TerminateProcess(process.get(), 1u);
    }
    WaitForSingleObject(process.get(), INFINITE);
    result.status = ProcessStatus::TimedOut;
  } else if (waitResult == WAIT_OBJECT_0) {
    DWORD exitCode = 0u;
    if (GetExitCodeProcess(process.get(), &exitCode)) {
      result.status = ProcessStatus::Exited;
      result.exitCode = static_cast<int>(exitCode);
    } else {
      result.status = ProcessStatus::InternalError;
      result.errorMessage =
          "GetExitCodeProcess failed: " + windowsErrorMessage(GetLastError());
    }
  } else {
    if (job.get() != nullptr) {
      TerminateJobObject(job.get(), 1u);
    } else {
      TerminateProcess(process.get(), 1u);
    }
    WaitForSingleObject(process.get(), INFINITE);
    result.status = ProcessStatus::InternalError;
    result.errorMessage =
        "waiting for process failed: " + windowsErrorMessage(GetLastError());
  }

  job.reset();
  stdoutThread.join();
  stderrThread.join();
  setElapsed(result, started);
  return result;
}

#else

class UniqueFd {
public:
  UniqueFd() = default;
  explicit UniqueFd(int fd) : fd_(fd) {}
  UniqueFd(const UniqueFd &) = delete;
  UniqueFd &operator=(const UniqueFd &) = delete;
  UniqueFd(UniqueFd &&other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
  UniqueFd &operator=(UniqueFd &&other) noexcept {
    if (this != &other) {
      reset(std::exchange(other.fd_, -1));
    }
    return *this;
  }
  ~UniqueFd() { reset(); }

  [[nodiscard]] int get() const noexcept { return fd_; }
  [[nodiscard]] int release() noexcept { return std::exchange(fd_, -1); }
  void reset(int fd = -1) noexcept {
    if (fd_ >= 0) {
      close(fd_);
    }
    fd_ = fd;
  }

private:
  int fd_ = -1;
};

[[nodiscard]] bool createPipe(UniqueFd &readFd, UniqueFd &writeFd,
                              const bool closeWriteOnExec,
                              std::string &errorMessage) {
  int descriptors[2]{};
  if (pipe(descriptors) != 0) {
    errorMessage = "pipe failed: " + std::string(std::strerror(errno));
    return false;
  }
  readFd.reset(descriptors[0]);
  writeFd.reset(descriptors[1]);
  if (closeWriteOnExec && fcntl(writeFd.get(), F_SETFD, FD_CLOEXEC) == -1) {
    errorMessage =
        "fcntl(FD_CLOEXEC) failed: " + std::string(std::strerror(errno));
    return false;
  }
  return true;
}

void readPipe(const int fd, std::string &output) {
  UniqueFd owned(fd);
  std::array<char, 16u * 1024u> buffer{};
  for (;;) {
    const ssize_t bytesRead = read(owned.get(), buffer.data(), buffer.size());
    if (bytesRead > 0) {
      output.append(buffer.data(), static_cast<size_t>(bytesRead));
      continue;
    }
    if (bytesRead < 0 && errno == EINTR) {
      continue;
    }
    break;
  }
}

enum class SpawnFailureStage : uint8_t {
  ChangeDirectory,
  DuplicateHandle,
  Exec
};
struct SpawnFailure {
  SpawnFailureStage stage{};
  int error = 0;
};

[[noreturn]] void reportSpawnFailure(const int fd,
                                     const SpawnFailureStage stage) {
  const SpawnFailure failure{stage, errno};
  (void)write(fd, &failure, sizeof(failure));
  _exit(127);
}

[[nodiscard]] std::string spawnFailureMessage(const SpawnFailure &failure) {
  std::string stage;
  switch (failure.stage) {
  case SpawnFailureStage::ChangeDirectory:
    stage = "chdir";
    break;
  case SpawnFailureStage::DuplicateHandle:
    stage = "dup2";
    break;
  case SpawnFailureStage::Exec:
    stage = "exec";
    break;
  }
  return stage + " failed: " + std::string(std::strerror(failure.error));
}

[[nodiscard]] bool waitForChild(const pid_t pid,
                                const std::chrono::milliseconds timeout,
                                int &status, bool &timedOut,
                                std::string &errorMessage) {
  timedOut = false;
  if (timeout == (std::chrono::milliseconds::max)()) {
    for (;;) {
      const pid_t waited = waitpid(pid, &status, 0);
      if (waited == pid) {
        return true;
      }
      if (waited < 0 && errno == EINTR) {
        continue;
      }
      errorMessage = "waitpid failed: " + std::string(std::strerror(errno));
      return false;
    }
  }

  const Clock::time_point deadline =
      Clock::now() + std::max(timeout, std::chrono::milliseconds::zero());
  for (;;) {
    const pid_t waited = waitpid(pid, &status, WNOHANG);
    if (waited == pid) {
      return true;
    }
    if (waited < 0 && errno != EINTR) {
      errorMessage = "waitpid failed: " + std::string(std::strerror(errno));
      return false;
    }
    if (Clock::now() >= deadline) {
      timedOut = true;
      kill(-pid, SIGKILL);
      kill(pid, SIGKILL);
      while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
      }
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

ProcessResult runProcessPlatform(const ProcessCommand &command,
                                 const ProcessOptions &options,
                                 const Clock::time_point started) {
  ProcessResult result{};
  if (command.executable.empty()) {
    result.errorMessage = "process executable is empty";
    setElapsed(result, started);
    return result;
  }

  std::vector<std::string> argumentStorage;
  argumentStorage.reserve(command.arguments.size() + 1u);
  argumentStorage.push_back(command.executable.string());
  argumentStorage.insert(argumentStorage.end(), command.arguments.begin(),
                         command.arguments.end());
  std::vector<char *> argv;
  argv.reserve(argumentStorage.size() + 1u);
  for (std::string &argument : argumentStorage) {
    argv.push_back(argument.data());
  }
  argv.push_back(nullptr);

  UniqueFd stdoutRead;
  UniqueFd stdoutWrite;
  UniqueFd stderrRead;
  UniqueFd stderrWrite;
  UniqueFd spawnRead;
  UniqueFd spawnWrite;
  if (!createPipe(stdoutRead, stdoutWrite, false, result.errorMessage) ||
      !createPipe(stderrRead, stderrWrite, false, result.errorMessage) ||
      !createPipe(spawnRead, spawnWrite, true, result.errorMessage)) {
    setElapsed(result, started);
    return result;
  }

  const pid_t pid = fork();
  if (pid < 0) {
    result.errorMessage = "fork failed: " + std::string(std::strerror(errno));
    setElapsed(result, started);
    return result;
  }
  if (pid == 0) {
    stdoutRead.reset();
    stderrRead.reset();
    spawnRead.reset();
    (void)setpgid(0, 0);
    if (options.workingDirectory.has_value() &&
        chdir(options.workingDirectory->c_str()) != 0) {
      reportSpawnFailure(spawnWrite.get(), SpawnFailureStage::ChangeDirectory);
    }
    if (dup2(stdoutWrite.get(), STDOUT_FILENO) < 0 ||
        dup2(stderrWrite.get(), STDERR_FILENO) < 0) {
      reportSpawnFailure(spawnWrite.get(), SpawnFailureStage::DuplicateHandle);
    }
    stdoutWrite.reset();
    stderrWrite.reset();
    execvp(argv[0], argv.data());
    reportSpawnFailure(spawnWrite.get(), SpawnFailureStage::Exec);
  }

  (void)setpgid(pid, pid);
  stdoutWrite.reset();
  stderrWrite.reset();
  spawnWrite.reset();

  SpawnFailure failure{};
  ssize_t failureBytes = 0;
  do {
    failureBytes = read(spawnRead.get(), &failure, sizeof(failure));
  } while (failureBytes < 0 && errno == EINTR);
  spawnRead.reset();
  if (failureBytes > 0) {
    int ignoredStatus = 0;
    while (waitpid(pid, &ignoredStatus, 0) < 0 && errno == EINTR) {
    }
    result.errorMessage = spawnFailureMessage(failure);
    setElapsed(result, started);
    return result;
  }
  if (failureBytes < 0) {
    const int readError = errno;
    kill(-pid, SIGKILL);
    kill(pid, SIGKILL);
    int ignoredStatus = 0;
    while (waitpid(pid, &ignoredStatus, 0) < 0 && errno == EINTR) {
    }
    result.status = ProcessStatus::InternalError;
    result.errorMessage =
        "reading spawn status failed: " + std::string(std::strerror(readError));
    setElapsed(result, started);
    return result;
  }

  std::thread stdoutThread;
  std::thread stderrThread;
  try {
    stdoutThread = std::thread(readPipe, stdoutRead.release(),
                               std::ref(result.standardOutput));
    stderrThread = std::thread(readPipe, stderrRead.release(),
                               std::ref(result.standardError));
  } catch (const std::system_error &error) {
    kill(-pid, SIGKILL);
    kill(pid, SIGKILL);
    int ignoredStatus = 0;
    while (waitpid(pid, &ignoredStatus, 0) < 0 && errno == EINTR) {
    }
    if (stdoutThread.joinable()) {
      stdoutThread.join();
    }
    if (stderrThread.joinable()) {
      stderrThread.join();
    }
    result.status = ProcessStatus::InternalError;
    result.errorMessage =
        "starting output capture failed: " + std::string(error.what());
    setElapsed(result, started);
    return result;
  }

  int childStatus = 0;
  bool timedOut = false;
  if (!waitForChild(pid, options.timeout, childStatus, timedOut,
                    result.errorMessage)) {
    kill(-pid, SIGKILL);
    kill(pid, SIGKILL);
    while (waitpid(pid, &childStatus, 0) < 0 && errno == EINTR) {
    }
    result.status = ProcessStatus::InternalError;
  } else if (timedOut) {
    result.status = ProcessStatus::TimedOut;
  } else {
    result.status = ProcessStatus::Exited;
    if (WIFEXITED(childStatus)) {
      result.exitCode = WEXITSTATUS(childStatus);
    } else if (WIFSIGNALED(childStatus)) {
      result.terminationSignal = WTERMSIG(childStatus);
      result.exitCode = 128 + *result.terminationSignal;
    } else {
      result.status = ProcessStatus::InternalError;
      result.errorMessage = "child ended with an unknown wait status";
    }
  }

  stdoutThread.join();
  stderrThread.join();
  setElapsed(result, started);
  return result;
}

#endif

} // namespace

ProcessResult runProcess(const ProcessCommand &command,
                         const ProcessOptions &options) {
  return runProcessPlatform(command, options, Clock::now());
}

} // namespace nuri::tools::core

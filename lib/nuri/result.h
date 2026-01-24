#pragma once

#include "defines.h"
#include "pch.h"

namespace nuri {

template <typename R, typename E> class Result {
public:
  Result(const R &value) : hasValue_(true) { new (&storage_.value) R(value); }

  Result(R &&value) : hasValue_(true) {
    new (&storage_.value) R(std::move(value));
  }

  Result(const E &error) : hasValue_(false) { new (&storage_.error) E(error); }

  Result(E &&error) : hasValue_(false) {
    new (&storage_.error) E(std::move(error));
  }

  Result(const Result &other) : hasValue_(other.hasValue_) {
    if (hasValue_) {
      new (&storage_.value) R(other.storage_.value);
    } else {
      new (&storage_.error) E(other.storage_.error);
    }
  }

  Result(Result &&other) noexcept : hasValue_(other.hasValue_) {
    if (hasValue_) {
      new (&storage_.value) R(std::move(other.storage_.value));
    } else {
      new (&storage_.error) E(std::move(other.storage_.error));
    }
    other.hasValue_ = true;
  }

  Result &operator=(const Result &other) {
    if (this != &other) {
      destroy();
      hasValue_ = other.hasValue_;
      if (hasValue_) {
        new (&storage_.value) R(other.storage_.value);
      } else {
        new (&storage_.error) E(other.storage_.error);
      }
    }
    return *this;
  }

  Result &operator=(Result &&other) noexcept {
    if (this != &other) {
      destroy();
      hasValue_ = other.hasValue_;
      if (hasValue_) {
        new (&storage_.value) R(std::move(other.storage_.value));
      } else {
        new (&storage_.error) E(std::move(other.storage_.error));
      }
      other.hasValue_ = true;
    }
    return *this;
  }

  ~Result() { destroy(); }

  [[nodiscard]] bool hasValue() const noexcept { return hasValue_; }
  [[nodiscard]] bool hasError() const noexcept { return !hasValue_; }
  [[nodiscard]] explicit operator bool() const noexcept { return hasValue_; }

  [[nodiscard]] R &value() & {
    if (!hasValue_) {
      throw std::runtime_error("Result does not contain a value");
    }
    return storage_.value;
  }

  [[nodiscard]] const R &value() const & {
    if (!hasValue_) {
      throw std::runtime_error("Result does not contain a value");
    }
    return storage_.value;
  }

  [[nodiscard]] R &&value() && {
    if (!hasValue_) {
      throw std::runtime_error("Result does not contain a value");
    }
    return std::move(storage_.value);
  }

  // Access error (throws if value)
  [[nodiscard]] E &error() & {
    if (hasValue_) {
      throw std::runtime_error("Result does not contain an error");
    }
    return storage_.error;
  }

  [[nodiscard]] const E &error() const & {
    if (hasValue_) {
      throw std::runtime_error("Result does not contain an error");
    }
    return storage_.error;
  }

  [[nodiscard]] E &&error() && {
    if (hasValue_) {
      throw std::runtime_error("Result does not contain an error");
    }
    return std::move(storage_.error);
  }

  [[nodiscard]] R &operator*() & { return value(); }
  [[nodiscard]] const R &operator*() const & { return value(); }
  [[nodiscard]] R &&operator*() && { return std::move(value()); }
  [[nodiscard]] R *operator->() { return &value(); }
  [[nodiscard]] const R *operator->() const { return &value(); }

  [[nodiscard]] static inline Result<R, E> makeResult(const R &value) {
    return Result<R, E>(value);
  }

  [[nodiscard]] static inline Result<R, E> makeResult(R &&value) {
    return Result<R, E>(std::forward<R>(value));
  }

  [[nodiscard]] static inline Result<R, E> makeError(const E &error) {
    return Result<R, E>(error);
  }

  [[nodiscard]] static inline Result<R, E> makeError(E &&error) {
    return Result<R, E>(std::forward<E>(error));
  }

private:
  void destroy() {
    if (hasValue_) {
      storage_.value.~R();
    } else {
      storage_.error.~E();
    }
  }

  union Storage {
    R value;
    E error;

    Storage() {}
    ~Storage() {}
  };

  Storage storage_;
  bool hasValue_;
};
} // namespace nuri
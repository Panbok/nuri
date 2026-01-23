#include "nuri/nuri.h"

class NuriApplication : public nuri::Application {
public:
  NuriApplication(const std::string &title, std::int32_t width,
                  std::int32_t height)
      : nuri::Application(title, width, height) {}

  void onUpdate(float deltaTime) override {}
  void onRender() override {}
  void onShutdown() override {}
};

int main() {
  NuriApplication app{"Nuri", 960, 540};
  app.run();
  return 0;
}

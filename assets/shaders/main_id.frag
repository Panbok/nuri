#include "common.sp"

layout(location = 9) flat in uint inInstanceId;

layout(location = 0) out uint outObjectId;

// 1-based object ID: 0 is reserved as sentinel. Saturate to avoid wrap when inInstanceId == UINT_MAX.
void main() {
  outObjectId = (inInstanceId >= 0xFFFFFFFFu) ? 0xFFFFFFFFu : (inInstanceId + 1u);
}

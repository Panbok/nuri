#include "common.sp"

layout(location = 9) flat in uint inInstanceId;

layout(location = 0) out uint outObjectId;

void main() { outObjectId = inInstanceId + 1u; }

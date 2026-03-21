#pragma once

#include "nuri/core/result.h"
#include "nuri/resources/storage/material/material_binary_codec.h"
#include "nuri/resources/storage/material/material_binary_serializer.h"

namespace nuri::material_binary_schema_codec {

void writeSceneMaterialRecord(material_binary_codec::Writer &writer,
                              const SceneMaterialRecord &record);

[[nodiscard]] Result<SceneMaterialRecord, MaterialBinaryDeserializeError>
readSceneMaterialRecord(material_binary_codec::Reader &reader);

} // namespace nuri::material_binary_schema_codec

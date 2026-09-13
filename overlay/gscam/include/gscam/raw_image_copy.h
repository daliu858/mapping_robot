#ifndef GSCAM_RAW_IMAGE_COPY_H
#define GSCAM_RAW_IMAGE_COPY_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gscam {

// Copy the active pixels of a packed single-plane image into a tightly packed
// ROS image buffer.  source_stride may include padding at the end of each row.
// The function validates every size calculation before touching source memory.
bool copyRawImageRows(const std::uint8_t* source,
                      std::size_t source_size,
                      std::size_t source_offset,
                      std::size_t source_stride,
                      std::size_t row_bytes,
                      std::size_t height,
                      std::vector<std::uint8_t>* destination,
                      std::string* error);

}  // namespace gscam

#endif  // GSCAM_RAW_IMAGE_COPY_H

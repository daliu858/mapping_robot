#include <gscam/raw_image_copy.h>

#include <algorithm>
#include <limits>

namespace gscam {
namespace {

bool fail(const char* message,
          std::vector<std::uint8_t>* destination,
          std::string* error) {
  if (destination) {
    destination->clear();
  }
  if (error) {
    *error = message;
  }
  return false;
}

}  // namespace

bool copyRawImageRows(const std::uint8_t* source,
                      std::size_t source_size,
                      std::size_t source_offset,
                      std::size_t source_stride,
                      std::size_t row_bytes,
                      std::size_t height,
                      std::vector<std::uint8_t>* destination,
                      std::string* error) {
  if (!destination) {
    return fail("destination is null", destination, error);
  }
  destination->clear();
  if (error) {
    error->clear();
  }
  if (!source) {
    return fail("source is null", destination, error);
  }
  if (row_bytes == 0 || height == 0) {
    return fail("row size and height must be positive", destination, error);
  }
  if (source_stride < row_bytes) {
    return fail("source stride is smaller than the active row", destination,
                error);
  }
  if (source_offset > source_size) {
    return fail("source offset is outside the buffer", destination, error);
  }

  const std::size_t max_size = std::numeric_limits<std::size_t>::max();
  if (height > max_size / row_bytes) {
    return fail("destination size overflows size_t", destination, error);
  }
  if (height > 1 &&
      (height - 1) > (max_size - row_bytes) / source_stride) {
    return fail("source extent overflows size_t", destination, error);
  }
  const std::size_t required_source_bytes =
      (height - 1) * source_stride + row_bytes;
  if (required_source_bytes > source_size - source_offset) {
    return fail("source buffer does not contain every active row", destination,
                error);
  }

  destination->resize(height * row_bytes);
  for (std::size_t row = 0; row < height; ++row) {
    const std::uint8_t* row_begin =
        source + source_offset + row * source_stride;
    std::copy(row_begin, row_begin + row_bytes,
              destination->begin() + row * row_bytes);
  }
  return true;
}

}  // namespace gscam

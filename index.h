
#include <cstddef>
#include <string>
#include <vector>
#include <fstream>
#include "parameters.h"


class Index {
 public:
  explicit Index(const size_t dimension, const size_t n, Metric metric);
  size_t nd_;

  virtual ~Index();

  virtual void Build(size_t n, const float *data, const Parameters &parameters) = 0;

  virtual void Search(
      const float *query,
      const float *x,
      size_t k,
      const Parameters &parameters,
      unsigned *indices) = 0;

  virtual void Save(const char *filename) = 0;

  virtual void Load(const char *filename) = 0;

  inline bool HasBuilt() const { return has_built; }

  inline size_t GetDimension() const { return dimension_; };

  inline size_t GetSizeOfDataset() const { return nd_; }

  inline const float *GetDataset() const { return data_; }
 protected:
  const size_t dimension_;
  const float *data_;
  
  bool has_built;
  
};


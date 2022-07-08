
#include <index.h>

Index::Index(const size_t dimension, const size_t n, Metric metric = L2)
  : dimension_ (dimension), nd_(n), has_built(false) {
    
}

Index::~Index() {}


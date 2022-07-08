
#include "util.h"
#include "parameters.h"
#include "neighbor.h"
#include "index.h"
#include <cassert>
#include <unordered_map>
#include <string>
#include <sstream>
#include <boost/dynamic_bitset.hpp>
#include <stack>
#include <set>
#include <map>
#include <bits/stdc++.h> 




class IndexTauMNG : public Index {
 public:
  explicit IndexTauMNG(const size_t dimension, const size_t n, Metric m, Index *initializer);


  virtual ~IndexTauMNG();

  virtual void Save(const char *filename)override;
  virtual void Load(const char *filename)override;


  virtual void Build(size_t n, const float *data, const Parameters &parameters) override;




  void Search(const float *query, const float *x, size_t K,
                      const Parameters &parameters, unsigned *indices,
                      std::vector<std::vector<int> > & perm_list,
                      int gtNN,
                      float * trans_q,
                      float * trans_data);

void Search_QEO(const float *query, const float *x, size_t K,
                      const Parameters &parameters, unsigned *indices,
                      std::vector<std::vector<int> > & perm_list,
                      int gtNN,
                      float * trans_q,
                      float * trans_data);

void Search_QEO_PDP_PII(const float *query, const float *x, size_t K,
                      const Parameters &parameters, unsigned *indices, 
                      std::vector<std::vector<float> > & uT2,
                      float * trans_q,
                      float * trans_data,
                      float * data_step_sum,
                      std::vector<float> & trans_q_steps);

  

  float eval_recall(std::vector<std::vector<unsigned> > query_res, 
        std::vector<std::vector<int> > gts,
        int K);



  protected:
    typedef std::vector<std::vector<unsigned > > CompactGraph;
    typedef std::vector<SimpleNeighbors > LockGraph;
    typedef std::vector<nhood> KNNGraph;


    CompactGraph final_graph_;

    Index *initializer_;
    void init_graph(const Parameters &parameters);
    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset);
    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        boost::dynamic_bitset<>& flags,
        std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset);
    //void add_cnn(unsigned des, Neighbor p, unsigned range, LockGraph& cut_graph_);
    void InterInsert(unsigned n, unsigned range, std::vector<std::mutex>& locks, SimpleNeighbor* cut_graph_);
    void sync_prune(unsigned q, std::vector<Neighbor>& pool, const Parameters &parameter, boost::dynamic_bitset<>& flags, SimpleNeighbor* cut_graph_);
    void Link(const Parameters &parameters, SimpleNeighbor* cut_graph_);
    void Load_nn_graph(const char *filename);
    void tree_grow(const Parameters &parameter);
    void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);
    void findroot(boost::dynamic_bitset<> &flag, unsigned &root, const Parameters &parameter);


  private:
    unsigned width;
    unsigned ep_;
    std::vector<std::mutex> locks;
    char* opt_graph_;
    size_t node_size;
    size_t data_len;
    size_t neighbor_len;
    KNNGraph nnd_graph;

  public:
    std::vector<std::vector<float> > final_graph_edge_length_;
    double NDC;
    int hops;
    float ang;
    float avg_tau;
    double comp_amount;
    float tau;
};




#include "index_taumng.h"

#include <omp.h>
#include <bitset>
#include <chrono>
#include <cmath>
#include <boost/dynamic_bitset.hpp>
#include "parameters.h"
#include <set>
#include <bits/stdc++.h> 



#define _CONTROL_NUM 100


IndexTauMNG::IndexTauMNG(const size_t dimension, const size_t n, Metric m,
                   Index *initializer)
    : Index(dimension, n, m), initializer_{initializer} {}

IndexTauMNG::~IndexTauMNG() {}


void IndexTauMNG::Save(const char *filename) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  assert(final_graph_.size() == nd_);

  out.write((char *)&width, sizeof(unsigned));
  out.write((char *)&ep_, sizeof(unsigned));
  for (unsigned i = 0; i < nd_; i++) {
    unsigned GK = (unsigned)final_graph_[i].size();
    out.write((char *)&GK, sizeof(unsigned));
    out.write((char *)final_graph_[i].data(), GK * sizeof(unsigned));
  }

  out.close();

  int edge_num = 0;
  for (int i = 0; i < final_graph_.size(); i++) {
    edge_num += final_graph_[i].size();
  }
}



void IndexTauMNG::Load(const char *filename) {
  std::ifstream in(filename, std::ios::binary);
  in.read((char *)&width, sizeof(unsigned));
  in.read((char *)&ep_, sizeof(unsigned));

  unsigned cc = 0;
  int edge_num = 0;
  while (!in.eof()) {
    unsigned k;
    in.read((char *)&k, sizeof(unsigned));
    if (in.eof()) break;
    cc += k;
    std::vector<unsigned> tmp(k);
    in.read((char *)tmp.data(), k * sizeof(unsigned));
    edge_num += tmp.size();
    final_graph_.push_back(tmp);
  }
  cc /= nd_;
}


void IndexTauMNG::Load_nn_graph(const char *filename) {
  std::ifstream in(filename, std::ios::binary);
  unsigned k;
  in.read((char *)&k, sizeof(unsigned));
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  size_t num = (unsigned)(fsize / (k + 1) / 4);
  in.seekg(0, std::ios::beg);

  final_graph_.resize(num);
  final_graph_.reserve(num);
  unsigned kk = (k + 3) / 4 * 4;
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    final_graph_[i].resize(k);
    final_graph_[i].reserve(kk);
    in.read((char *)final_graph_[i].data(), k * sizeof(unsigned));
  }
  in.close();

  int edge_num = 0;
  for (int i = 0; i < final_graph_.size(); i++) {
    edge_num += final_graph_[i].size();
  }
}



void IndexTauMNG::get_neighbors(const float *query, const Parameters &parameter,
                             std::vector<Neighbor> &retset,
                             std::vector<Neighbor> &fullset) {
  unsigned L = parameter.Get<unsigned>("L");
  retset.resize(L + 1);
  std::vector<unsigned> init_ids(L);

  boost::dynamic_bitset<> flags{nd_, 0};
  L = 0;
  for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size(); i++) {
    init_ids[i] = final_graph_[ep_][i];
    flags[init_ids[i]] = true;
    L++;
  }
  while (L < init_ids.size()) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    init_ids[L] = id;
    L++;
    flags[id] = true;
  }

  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
                                    (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    L++;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;

        float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                        (unsigned)dimension_);
        Neighbor nn(id, dist, true);
        fullset.push_back(nn);
        if (dist >= retset[L - 1].distance) continue;
        int r = InsertIntoPool(retset.data(), L, nn); //r is the pos of nn in L. L is sorted. If insertion fail, r = L+1

        if (L + 1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k) 
      k = nk;
    else
      ++k;
  }
}

void IndexTauMNG::get_neighbors(const float *query, const Parameters &parameter,
                             boost::dynamic_bitset<> &flags,
                             std::vector<Neighbor> &retset,
                             std::vector<Neighbor> &fullset) {
  unsigned L = parameter.Get<unsigned>("L");

  retset.resize(L + 1);
  std::vector<unsigned> init_ids(L);

  L = 0;
  for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size(); i++) {
    init_ids[i] = final_graph_[ep_][i];
    flags[init_ids[i]] = true;
    L++;
  }
  while (L < init_ids.size()) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    init_ids[L] = id;
    L++;
    flags[id] = true;
  }

  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
                                    (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    fullset.push_back(retset[i]);
    L++;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;

        float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                        (unsigned)dimension_);
        Neighbor nn(id, dist, true);
        fullset.push_back(nn);
        if (dist >= retset[L - 1].distance) continue;
        int r = InsertIntoPool(retset.data(), L, nn);

        if (L + 1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
}



void IndexTauMNG::init_graph(const Parameters &parameters) {
  float *center = new float[dimension_];
  for (unsigned j = 0; j < dimension_; j++) center[j] = 0;
  for (unsigned i = 0; i < nd_; i++) {
    for (unsigned j = 0; j < dimension_; j++) {
      center[j] += data_[i * dimension_ + j];
    }
  }
  for (unsigned j = 0; j < dimension_; j++) {
    center[j] /= nd_;
  }
  std::vector<Neighbor> tmp, pool;
  ep_ = rand() % nd_;  
  get_neighbors(center, parameters, tmp, pool);
  ep_ = tmp[0].id; 
}



void IndexTauMNG::sync_prune(unsigned q, std::vector<Neighbor> &pool,
                          const Parameters &parameter,
                          boost::dynamic_bitset<> &flags,
                          SimpleNeighbor *cut_graph_) {
  unsigned range = parameter.Get<unsigned>("R");
  unsigned maxc = parameter.Get<unsigned>("C");
  width = range;
  unsigned start = 0;

  for (unsigned nn = 0; nn < final_graph_[q].size(); nn++) {
    unsigned id = final_graph_[q][nn];
    if (flags[id]) continue;
    float dist =
        distance_->compare(data_ + dimension_ * (size_t)q,
                           data_ + dimension_ * (size_t)id, (unsigned)dimension_);
    pool.push_back(Neighbor(id, dist, true));
  }

  std::sort(pool.begin(), pool.end());
  std::vector<Neighbor> result;
  if (pool[start].id == q) start++;
  result.push_back(pool[start]);

  while (result.size() < range && (++start) < pool.size() && start < maxc) {
    auto &p = pool[start];
    bool occlude = false;
    for (unsigned t = 0; t < result.size(); t++) {
      if (p.id == result[t].id) {
        occlude = true;
        break;
      }
      float djk = distance_->compare(data_ + dimension_ * (size_t)result[t].id,
                                     data_ + dimension_ * (size_t)p.id,
                                     (unsigned)dimension_);
      float dist_existNeigh_p = djk;
      float dist_q_p = p.distance;
      if (dist_existNeigh_p < dist_q_p - 3*tau) { 
        occlude = true;
        break;
      }
    }
    if (!occlude) result.push_back(p);
  }

  SimpleNeighbor *des_pool = cut_graph_ + (size_t)q * (size_t)range;
  for (size_t t = 0; t < result.size(); t++) {
    des_pool[t].id = result[t].id;
    des_pool[t].distance = result[t].distance;
  }
  if (result.size() < range) {
    des_pool[result.size()].distance = -1;
  }
}



void IndexTauMNG::InterInsert(unsigned n, unsigned range,
                           std::vector<std::mutex> &locks,
                           SimpleNeighbor *cut_graph_) {

  SimpleNeighbor *src_pool = cut_graph_ + (size_t)n * (size_t)range;
  for (size_t i = 0; i < range; i++) {
    if (src_pool[i].distance == -1) break;

    SimpleNeighbor sn(n, src_pool[i].distance);
    size_t des = src_pool[i].id;
    SimpleNeighbor *des_pool = cut_graph_ + des * (size_t)range;

    std::vector<SimpleNeighbor> temp_pool;
    int dup = 0;
    {
      LockGuard guard(locks[des]);
      for (size_t j = 0; j < range; j++) {
        if (des_pool[j].distance == -1) break;
        if (n == des_pool[j].id) {
          dup = 1;
          break;
        }
        temp_pool.push_back(des_pool[j]);
      }
    }
    if (dup) continue;

    temp_pool.push_back(sn);
    if (temp_pool.size() > range) {
      std::vector<SimpleNeighbor> result;
      unsigned start = 0;
      std::sort(temp_pool.begin(), temp_pool.end());
      result.push_back(temp_pool[start]);
      while (result.size() < range && (++start) < temp_pool.size()) {
        auto &p = temp_pool[start];
        bool occlude = false;
        for (unsigned t = 0; t < result.size(); t++) {
          if (p.id == result[t].id) {
            occlude = true;
            break;
          }
          float djk = distance_->compare(data_ + dimension_ * (size_t)result[t].id,
                                         data_ + dimension_ * (size_t)p.id,
                                         (unsigned)dimension_);
          
          float dist_existNeigh_p = djk;
          float dist_q_p = p.distance;
          if (dist_existNeigh_p < dist_q_p - 3*tau) {
            occlude = true;
            break;
          }
        }
        if (!occlude) result.push_back(p);
      }
      {
        LockGuard guard(locks[des]);
        for (unsigned t = 0; t < result.size(); t++) {
          des_pool[t] = result[t];
        }
      }
    } else {
      LockGuard guard(locks[des]);
      for (unsigned t = 0; t < range; t++) {
        if (des_pool[t].distance == -1) {
          des_pool[t] = sn;
          if (t + 1 < range) des_pool[t + 1].distance = -1;
          break;
        }
      }
    }
  }
}




void IndexTauMNG::Link(const Parameters &parameters, SimpleNeighbor *cut_graph_) {
  
  unsigned range = parameters.Get<unsigned>("R");
  std::vector<std::mutex> locks(nd_);

#pragma omp parallel
  {
    std::vector<Neighbor> pool, tmp;
    boost::dynamic_bitset<> flags{nd_, 0};
#pragma omp for schedule(dynamic, 100)
    for (unsigned n = 0; n < nd_; ++n) {
      pool.clear();
      tmp.clear();
      flags.reset();
      get_neighbors(data_ + dimension_ * n, parameters, flags, tmp, pool);
      sync_prune(n, pool, parameters, flags, cut_graph_);
    }
  }

#pragma omp for schedule(dynamic, 100)
  for (unsigned n = 0; n < nd_; ++n) {
    InterInsert(n, range, locks, cut_graph_); 
  }
}

void IndexTauMNG::Build(size_t n, const float *data, const Parameters &parameters) {
  std::string nn_graph_path = parameters.Get<std::string>("nn_graph_path");
  unsigned range = parameters.Get<unsigned>("R"); 
  Load_nn_graph(nn_graph_path.c_str());

  std::cout << "tau = " << tau << std::endl;

  data_ = data;
  init_graph(parameters);
  SimpleNeighbor *cut_graph_ = new SimpleNeighbor[nd_ * (size_t)range];
  Link(parameters, cut_graph_);
  final_graph_.resize(nd_);

  for (size_t i = 0; i < nd_; i++) {
    SimpleNeighbor *pool = cut_graph_ + i * (size_t)range;
    unsigned pool_size = 0;
    for (unsigned j = 0; j < range; j++) {
      if (pool[j].distance == -1) break;
      pool_size = j;
    }
    pool_size++;
    final_graph_[i].resize(pool_size);
    for (unsigned j = 0; j < pool_size; j++) {
      final_graph_[i][j] = pool[j].id;
    }
  }

  {
    unsigned max = 0, min = 1e6, avg = 0;
    for (size_t i = 0; i < nd_; i++) {
      auto size = final_graph_[i].size();
      max = max < size ? size : max;
      min = min > size ? size : min;
      avg += size;
    }
    avg /= 1.0 * nd_;
    printf("Degree Statistics: Max = %d, Min = %d, Avg = %d\n", max, min, avg);
  }

  tree_grow(parameters);

  unsigned max = 0, min = 1e6, avg = 0;
  for (size_t i = 0; i < nd_; i++) {
    auto size = final_graph_[i].size();
    max = max < size ? size : max;
    min = min > size ? size : min;
    avg += size;
  }
  avg /= 1.0 * nd_;
  printf("Degree Statistics: Max = %d, Min = %d, Avg = %d\n", max, min, avg);

  has_built = true;
}




bool sortbysec(const std::pair<int,float> &a,
              const std::pair<int,float> &b)
{
    return (a.second < b.second);
}

bool sortbyNeighbor(Neighbor &a,
              Neighbor &b)
{
    return (a.distance < b.distance);
}


void IndexTauMNG::Search_QEO(const float *query, const float *x, size_t K,
                      const Parameters &parameters, unsigned *indices,
                      std::vector<std::vector<int> > & perm_list,
                      int gtNN,
                      float * trans_q,
                      float * trans_data) {
                        
  const unsigned L = parameters.Get<unsigned>("L_search");
  unsigned stopL = parameters.Get<unsigned>("stopL");
  if (stopL > L) {
    stopL = L;
  }

  data_ = x;
  std::vector<Neighbor> retset(L + 1);
  std::vector<unsigned> init_ids(L);
  boost::dynamic_bitset<> flags{nd_, 0};
  float kperc = parameters.Get<float>("kperc");

  unsigned tmp_l = 0;
  init_ids[tmp_l] = ep_; flags[ep_] = true; tmp_l++;
  


  for (unsigned i = 0; i < tmp_l; i++) {
    unsigned id = init_ids[i];
    float dist = distance_->compare_by_loop(data_ + dimension_ * id, query, (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    NDC++;
  }

  for (unsigned i = tmp_l; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    // float dist = distance_->compare_by_loop(data_ + dimension_ * id, query, (unsigned)dimension_);
    // NDC++;
    // retset[i] = Neighbor(id, dist, true);
    retset[i] = Neighbor(-1, 100000000.0, false);
  }


  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  int hop = 0;
 

  while (k < (int) stopL) {
    int nk = stopL;

    
    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      if (k < kperc * L) {
        for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
          unsigned id = final_graph_[n][m];
          if (flags[id]) continue;
          flags[id] = 1;

          float dist = 0.0;
          for (int oo = 0; oo < dimension_; oo++) {
            dist += (query[oo] - (data_ + dimension_ * id)[oo]) * (query[oo] - (data_ + dimension_ * id)[oo]);
            comp_amount++;
          }
          NDC++;

          if (dist >= retset[L - 1].distance) continue;
          Neighbor nn(id, dist, true);
          int r = InsertIntoPool(retset.data(), L, nn);

          if (r < nk) nk = r;
        }
      }
      else {
        
        float firstShortestDist = 1000000.0; int id_firstShortest = -1;
        float secondShortestDist = 2000000.0; int id_secondShortest = -1;

        for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
          unsigned id = final_graph_[n][m];
          if (flags[id]) continue;

          float partial_dist = 0.0;
          for (int oo = 0; oo < dimension_/2; oo++) {
            partial_dist += (trans_q[oo] - (trans_data + dimension_ * id)[oo]) * (trans_q[oo] - (trans_data + dimension_ * id)[oo]);
            comp_amount++;
          }

          NDC += 0.5;

          if (partial_dist < firstShortestDist) {
            secondShortestDist = firstShortestDist;
            id_secondShortest = id_firstShortest;

            firstShortestDist = partial_dist;
            id_firstShortest = m;
          }
          else {
            if (partial_dist < secondShortestDist) {
              secondShortestDist = partial_dist;
              id_secondShortest = m;
            }
          }
        }

        
        if (id_firstShortest != -1) {
          int id = final_graph_[n][id_firstShortest];
          flags[id] = 1;
          
          float dist = 0.0;
          for (int oo = 0; oo < dimension_; oo++) {
            dist += (query[oo] - (data_ + dimension_ * id)[oo]) * (query[oo] - (data_ + dimension_ * id)[oo]);
            comp_amount++;
          }
          NDC++;
          
          if (dist >= retset[L - 1].distance) continue;
          Neighbor nn(id, dist, true);
          int r = InsertIntoPool(retset.data(), L, nn);

          if (r < nk) nk = r;
        }

        if (id_secondShortest != -1) {
          int id = final_graph_[n][id_secondShortest];
          flags[id] = 1;
          
          float dist = 0.0;
          for (int oo = 0; oo < dimension_; oo++) {
            dist += (query[oo] - (data_ + dimension_ * id)[oo]) * (query[oo] - (data_ + dimension_ * id)[oo]);
            comp_amount++;
          }
          NDC++;
          
          if (dist >= retset[L - 1].distance) continue;
          Neighbor nn(id, dist, true);
          int r = InsertIntoPool(retset.data(), L, nn);

          if (r < nk) nk = r;
        }  
      }     
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
  for (size_t i = 0; i < K; i++) {
    indices[i] = retset[i].id;
  }

}




void IndexTauMNG::Search_QEO_PDP(const float *query, const float *x, size_t K,
                      const Parameters &parameters, unsigned *indices,
                      std::vector<std::vector<int> > & perm_list,
                      int gtNN,
                      float * trans_q,
                      float * trans_data) {
                        
  const unsigned L = parameters.Get<unsigned>("L_search");
  unsigned stopL = parameters.Get<unsigned>("stopL");
  if (stopL > L) {
    stopL = L;
  }

  data_ = x;
  std::vector<Neighbor> retset(L + 1);
  std::vector<unsigned> init_ids(L);
  boost::dynamic_bitset<> flags{nd_, 0};
  float kperc = parameters.Get<float>("kperc");

  unsigned tmp_l = 0;
  init_ids[tmp_l] = ep_; flags[ep_] = true; tmp_l++;


  for (unsigned i = 0; i < tmp_l; i++) {
    unsigned id = init_ids[i];
    float dist = distance_->compare_by_loop(data_ + dimension_ * id, query, (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    NDC++;
  }

  for (unsigned i = tmp_l; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    // float dist = distance_->compare_by_loop(data_ + dimension_ * id, query, (unsigned)dimension_);
    // NDC++;
    // retset[i] = Neighbor(id, dist, true);
    retset[i] = Neighbor(-1, 100000000.0, false);
  }


  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  int hop = 0;
 

  while (k < (int) stopL) {
    int nk = stopL;

    
    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;
      
      if (k < kperc * L) {
        for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
          unsigned id = final_graph_[n][m];
          if (flags[id]) continue;
          flags[id] = 1;

          float dist = 0.0;
          bool isBreak = false;
          for (int oo = 0; oo < dimension_; oo++) {
            dist += (trans_q[oo] - (trans_data + dimension_ * id)[oo]) * (trans_q[oo] - (trans_data + dimension_ * id)[oo]);
            comp_amount++;
            if (dist > retset[L - 1].distance) {
              isBreak = true;
              break;
            }
          }
          NDC++;
          if (isBreak) continue;
          
          if (dist >= retset[L - 1].distance) continue;
          Neighbor nn(id, dist, true);
          int r = InsertIntoPool(retset.data(), L, nn);
        
          if (r < nk) nk = r;
        }
      }
      else {

        float firstShortestDist = 1000000.0; int id_firstShortest = -1;
        float secondShortestDist = 2000000.0; int id_secondShortest = -1;

        for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
          unsigned id = final_graph_[n][m];
          if (flags[id]) continue;

          float partial_dist = 0.0;
          bool isBreak = false;
          for (int oo = 0; oo < dimension_/2; oo++) {
            partial_dist += (trans_q[oo] - (trans_data + dimension_ * id)[oo]) * (trans_q[oo] - (trans_data + dimension_ * id)[oo]);
            comp_amount++;
            if (partial_dist > retset[L - 1].distance) {
              isBreak = true;
              break;
            }
          }
          if (isBreak) {
            flags[id] = 1;
            continue;
          }          


          if (partial_dist < firstShortestDist) {
            secondShortestDist = firstShortestDist;
            id_secondShortest = id_firstShortest;

            firstShortestDist = partial_dist;
            id_firstShortest = m;
          }
          else {
            if (partial_dist < secondShortestDist) {
              secondShortestDist = partial_dist;
              id_secondShortest = m;
            }
          }
        }

        
        if (id_firstShortest != -1) {
          int id = final_graph_[n][id_firstShortest];
          flags[id] = 1;
          
          float dist = firstShortestDist;
          bool isBreak = false;
          for (int oo = dimension_/2; oo < dimension_; oo++) {
            dist += (trans_q[oo] - (trans_data + dimension_ * id)[oo]) * (trans_q[oo] - (trans_data + dimension_ * id)[oo]);
            if (dist > retset[L - 1].distance) {
              isBreak = true;
              break;
            }
          }
          NDC++;
          if (isBreak) {
            continue;
          }
          
          if (dist >= retset[L - 1].distance) continue;
          Neighbor nn(id, dist, true);
          int r = InsertIntoPool(retset.data(), L, nn);
      
          if (r < nk) nk = r;
        }

        if (id_secondShortest != -1) {
          int id = final_graph_[n][id_secondShortest];
          flags[id] = 1;
          
          float dist = secondShortestDist;
          bool isBreak = false;
          for (int oo = dimension_/2; oo < dimension_; oo++) {
            dist += (trans_q[oo] - (trans_data + dimension_ * id)[oo]) * (trans_q[oo] - (trans_data + dimension_ * id)[oo]);
            if (dist > retset[L - 1].distance) {
              isBreak = true;
              break;
            }
          }
          NDC++;
          if (isBreak) continue;
          
          if (dist >= retset[L - 1].distance) continue;
          Neighbor nn(id, dist, true);
          int r = InsertIntoPool(retset.data(), L, nn);
      
          if (r < nk) nk = r;
        }  
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
  for (size_t i = 0; i < K; i++) {
    indices[i] = retset[i].id;
  }

}



void IndexTauMNG::Search_QEO_PDP_PII(const float *query, const float *x, size_t K,
                      const Parameters &parameters, unsigned *indices, 
                      std::vector<std::vector<float> > & uT2,
                      float * trans_q,
                      float * trans_data,
                      float * data_step_sum,
                      std::vector<float> & trans_q_steps) {



  const unsigned L = parameters.Get<unsigned>("L_search");
  unsigned stopL = parameters.Get<unsigned>("stopL");
  if (stopL > L) {
    stopL = L;
  }
 
  int step_size = parameters.Get<int>("stepSize");
  float kperc = parameters.Get<float>("kperc");


  data_ = x;
  std::vector<Neighbor> retset(L + 1);
  std::vector<unsigned> init_ids(L);
  boost::dynamic_bitset<> flags{nd_, 0};

  unsigned tmp_l = 0;

  init_ids[tmp_l] = ep_; flags[ep_] = true; tmp_l++;
  


  int step_num = dimension_/step_size;
  if (dimension_/step_size < ((float)dimension_/step_size)) {
    step_num++;
  }

 
  for (unsigned i = 0; i < tmp_l; i++) {
    unsigned id = init_ids[i];
    float dist = distance_->compare_by_loop(data_ + dimension_ * id, query, (unsigned)dimension_);
    NDC++;
    not_break_comp_NDC++;
    not_break_comp_amount += 2*dimension_;
    retset[i] = Neighbor(id, dist, true);
    
  }


  for (unsigned i = tmp_l; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    retset[i] = Neighbor(-1, 100000000.0, false);
  }


  std::sort(retset.begin(), retset.begin() + L);

 
  int k = 0;
  int hop = 0;

  while (k < (int) stopL) {
    int nk = stopL;

    if (retset[k].flag) {

      retset[k].flag = false;
      unsigned n = retset[k].id;
      float dist_q_n = retset[k].distance;


      if (k < kperc * L) {
        for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
          
          unsigned id = final_graph_[n][m];
          if (flags[id]) continue;
          
          flags[id] = 1;

          float * neigh = trans_data + id * dimension_;
          float * neigh_step_sum = data_step_sum + id * step_num;
        
          
          float mydist = 0.0;
          float inner_prod = 0.0;
          bool isBreak = false;
          for (int i = 0; i < dimension_/step_size; i++) {
            int start = i*step_size; int end = i*step_size + step_size;
            for (int j = start; j < end; j++) {
              inner_prod += trans_q[j] * neigh[j];
            }
            mydist = (trans_q_steps[i] + neigh_step_sum[i] - 2*inner_prod);
            if (mydist > retset[L - 1].distance) {
              isBreak = true;
              break;
            }
          }
          if (isBreak) {
            continue;
          }
          if (dimension_/step_size < ((float)dimension_/step_size)) {
            for (int j = dimension_/step_size * step_size; j < dimension_; j++) {
              inner_prod += trans_q[j] * (trans_data + id * dimension_)[j];
            }
            mydist = (trans_q_steps[trans_q_steps.size()-1] + neigh_step_sum[trans_q_steps.size()-1] - 2*inner_prod);
          }
          
          
          float dist = mydist;
          if (dist >= retset[L - 1].distance) continue;

          Neighbor nn(id, dist, true);
          int r = InsertIntoPool(retset.data(), L, nn);

        
          if (r < nk) nk = r; //update the smallest updated position in retset
        }
      }
      else {

        float firstShortestDist = 1000000.0; int id_firstShortest = -1;
        float secondShortestDist = 2000000.0; int id_secondShortest = -1;

        for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
          unsigned id = final_graph_[n][m];
          if (flags[id]) continue;

          float * neigh = trans_data + id * dimension_;
          float * neigh_step_sum = data_step_sum + id * step_num;

          float partial_dist = 0.0;
          bool isBreak = false;
          float inner_prod = 0.0;
          for (int i = 0; i < (dimension_/2)/step_size; i++) {
            int start = i*step_size; int end = i*step_size + step_size;
            for (int j = start; j < end; j++) {
              inner_prod += trans_q[j] * neigh[j];
            }
            partial_dist = (trans_q_steps[i] + neigh_step_sum[i] - 2*inner_prod);
            if (partial_dist > retset[L - 1].distance) {
              isBreak = true;
              break;
            }
          }
          if (isBreak) {
            flags[id] = 1;
            continue;
          } 
          //note that (dimension_/2)/step_size * step_size may be < dimension_/2
          for (int j = (dimension_/2)/step_size * step_size; j < dimension_/2; j++) {
            partial_dist += (trans_q[j] - (trans_data + dimension_ * id)[j]) * (trans_q[j] - (trans_data + dimension_ * id)[j]);
            comp_amount += 1;
            if (partial_dist > retset[L - 1].distance) {
              isBreak = true;
              break;
            }
          }
          if (isBreak) {
            flags[id] = 1;
            continue;
          }
   

          if (partial_dist < firstShortestDist) {
            secondShortestDist = firstShortestDist;
            id_secondShortest = id_firstShortest;

            firstShortestDist = partial_dist;
            id_firstShortest = m;
          }
          else {
            if (partial_dist < secondShortestDist) {
              secondShortestDist = partial_dist;
              id_secondShortest = m;
            }
          }
        }

        
        if (id_firstShortest != -1) {
          int id = final_graph_[n][id_firstShortest];
          flags[id] = 1;
          
          float dist = firstShortestDist;
          bool isBreak = false;
          for (int oo = dimension_/2; oo < dimension_; oo++) {
            dist += (trans_q[oo] - (trans_data + dimension_ * id)[oo]) * (trans_q[oo] - (trans_data + dimension_ * id)[oo]);
            if (dist > retset[L - 1].distance) {
              isBreak = true;
              break;
            }
          }
          NDC++;
          if (isBreak) {
            continue;
          }
          
          if (dist >= retset[L - 1].distance) continue;
          Neighbor nn(id, dist, true);
          int r = InsertIntoPool(retset.data(), L, nn);
      
          if (r < nk) nk = r;
        }

        if (id_secondShortest != -1) {
          int id = final_graph_[n][id_secondShortest];
          flags[id] = 1;
          
          float dist = secondShortestDist;
          bool isBreak = false;
          for (int oo = dimension_/2; oo < dimension_; oo++) {
            dist += (trans_q[oo] - (trans_data + dimension_ * id)[oo]) * (trans_q[oo] - (trans_data + dimension_ * id)[oo]);
            if (dist > retset[L - 1].distance) {
              isBreak = true;
              break;
            }
          }
          NDC++;
          if (isBreak) continue;
          
          if (dist >= retset[L - 1].distance) continue;
          Neighbor nn(id, dist, true);
          int r = InsertIntoPool(retset.data(), L, nn);
      
          if (r < nk) nk = r;
        } 
      }
    }
   

    if (nk <= k)
      k = nk;
    else
      ++k;

    hop++;
  }
 

  for (size_t i = 0; i < K; i++) {
    indices[i] = retset[i].id;
  }
}




void IndexTauMNG::findroot(boost::dynamic_bitset<> &flag, unsigned &root,
                        const Parameters &parameter) {
  unsigned id = nd_;
  for (unsigned i = 0; i < nd_; i++) {
    if (flag[i] == false) {
      id = i;
      break;
    }
  }

  if (id == nd_) return;  // No Unlinked Node

  std::vector<Neighbor> tmp, pool;
  get_neighbors(data_ + dimension_ * id, parameter, tmp, pool);
  std::sort(pool.begin(), pool.end());

  unsigned found = 0;
  for (unsigned i = 0; i < pool.size(); i++) {
    if (flag[pool[i].id]) {
      root = pool[i].id;
      found = 1;
      break;
    }
  }
  if (found == 0) {
    while (true) {
      unsigned rid = rand() % nd_;
      if (flag[rid]) {
        root = rid;
        break;
      }
    }
  }
  final_graph_[root].push_back(id);
}


void IndexTauMNG::tree_grow(const Parameters &parameter) {
  unsigned root = ep_;
  boost::dynamic_bitset<> flags{nd_, 0};
  unsigned unlinked_cnt = 0;
  while (unlinked_cnt < nd_) {
    DFS(flags, root, unlinked_cnt);
    if (unlinked_cnt >= nd_) break;
    findroot(flags, root, parameter);
  }
  for (size_t i = 0; i < nd_; ++i) {
    if (final_graph_[i].size() > width) {
      width = final_graph_[i].size();
    }
  }
}


float IndexTauMNG::eval_recall(std::vector<std::vector<unsigned> > query_res, std::vector<std::vector<int> > gts, int K){
  float mean_recall=0;
  for(unsigned i=0; i<query_res.size(); i++){
    assert(query_res[i].size() <= gts[i].size());
    
    float recall = 0;
    std::set<unsigned> cur_query_res_set(query_res[i].begin(), query_res[i].end());
    std::set<int> cur_query_gt(gts[i].begin(), gts[i].begin()+K);
    
    for (std::set<unsigned>::iterator x = cur_query_res_set.begin(); x != cur_query_res_set.end(); x++) { 
      std::set<int>::iterator iter = cur_query_gt.find(*x);
      if (iter != cur_query_gt.end()) {
        recall++;
      }
    }
    recall = recall / query_res[i].size();
    mean_recall += recall;
  }
  mean_recall = (mean_recall / query_res.size());

  return mean_recall;
}




#!/usr/bin/env python3
# coding: utf-8

# # Adjacency Matrix

import numpy as np

from scipy.sparse import csr_matrix, lil_matrix, coo_matrix


class adj_matrix(object):
    def __init__(self, symmetric_indices=False):
        
        self._sym = symmetric_indices
        if not symmetric_indices:
            self._x_labels = []
            self._y_labels = []

            self._x_label_index={}
            self._y_label_index={}
        
        else:
            self._labels = []
            self._label_index={}
        
        self._x = []
        self._y = []
        self._data = []
        
        self._mat = None
        self._csr = None

        # for a TF-IDF like approach we need also a counter how frequently ingredients
        # and actions appear in documents. 

        self._current_document_labels = set()
        self._label_document_count = {}
        
        self._document_count = 0
        
        # building type dependend functions:
        self._build_funcs()
    
    def _get_ix(self, label):
        i = self._x_label_index.get(label)
        if i is None:
            i = len(self._x_labels)
            self._x_labels.append(label)
            self._x_label_index[label] = i
        return i
    
    def _get_iy(self, label):
        i = self._y_label_index.get(label)
        if i is None:
            i = len(self._y_labels)
            self._y_labels.append(label)
            self._y_label_index[label] = i
        return i
    
    def _get_i(self, label):
        i = self._label_index.get(label)
        if i is None:
            i = len(self._labels)
            self._labels.append(label)
            self._label_index[label] = i
        return i

    def _end_document(self):
        self._document_count += 1

        # adding all seen labels to our counter:
        for label in self._current_document_labels:
            self._label_document_count[label] += 1
        else:
            self._label_document_count[label] = 1
        
        self._current_document_labels = set()
    
    def apply_threshold(self, min_count=5):
        csr = self.get_csr()

        new_x = []
        new_y = []
        new_data = []

        for i in range(len(self._data)):
            if csr[self._x[i],self._y[i]] >= min_count:
                new_x.append(self._x[i])
                new_y.append(self._y[i])
                new_data.append(self._data[i])
        
        self._x = new_x
        self._y = new_y
        self._data = new_data

    
    def next_document(self):
        self._end_document()

    
    def add_entry(self, x, y, data):
        
        if self._sym:
            ix = self._get_i(x)
            iy = self._get_i(y)
        
        else:
            ix = self._get_ix(x)
            iy = self._get_iy(y)
        
        self._x.append(ix)
        self._y.append(iy)
        self._data.append(data)

        self._current_document_labels.add(x)
        self._current_document_labels.add(y)
    
    def compile(self):
        self._csr = self.get_csr()
        if self._sym:
            self._np_labels = np.array(self._labels)
        else:
            self._np_x_labels = np.array(self._x_labels)
            self._np_y_labels = np.array(self._y_labels)
        
    
    def compile_to_mat(self):
        if self._sym:
            sx = len(self._labels)
            sy = len(self._labels)
        else:
            sx = len(self._x_labels)
            sy = len(self._y_labels)
        
        self._mat = coo_matrix((self._data, (self._x, self._y)), shape=(sx,sy))
        return self._mat
    
    def get_csr(self):
        return self.compile_to_mat().tocsr()
    
    def get_labels(self):
        if self._sym:
            return self._labels
        return self._x_labels, self._y_labels
    
    def _build_funcs(self):
        
        def get_sym_adjacent(key):
            assert self._csr is not None
            
            c = self._csr
            
            index = self._label_index[key]
            i1 = c[index,:].nonzero()[1]
            i2 = c[:,index].nonzero()[0]

            i = np.concatenate((i1,i2))

            names = self._np_labels[i]

            counts = np.concatenate((c[index, i1].toarray().flatten(), c[i2, index].toarray().flatten()))

            s = np.argsort(-counts)

            return names[s], counts[s]
        
        def get_forward_adjacent(key):
            assert self._csr is not None
            
            c = self._csr
            
            index = self._x_label_index[key]
            i = c[index,:].nonzero()[1]

            names = self._np_y_labels[i]

            counts = c[index, i].toarray().flatten()

            s = np.argsort(-counts)

            return names[s], counts[s]
        
        def get_backward_adjacent(key):
            assert self._csr is not None
            
            c = self._csr
            
            index = self._y_label_index[key]
            i = c[:,index].nonzero()[0]

            
            names = self._np_x_labels[i]

            counts = c[i, index].toarray().flatten()

            s = np.argsort(-counts)

            return names[s], counts[s]
        
        # sum functions:
        def sym_sum(key):
            return np.sum(self.get_adjacent(key)[1])

        def fw_sum(key):
            return np.sum(self.get_forward_adjacent(key)[1])

        def bw_sum(key):
            return np.sum(self.get_backward_adjacent(key)[1])
        
        # normalization stuff:
        def fw_normalization_factor(key, quotient_func):
            assert self._csr is not None
            c = self._csr
            
            ia = self._x_label_index[key]

            occurances = c[ia,:].nonzero()[1]

            return 1. / quotient_func(c[ia,occurances].toarray())

        def bw_normalization_factor(key, quotient_func):
            assert self._csr is not None
            
            c = self._csr
            
            ib = m._y_label_index[key]

            occurances = c[:,ib].nonzero()[0]

            return 1. / quotient_func(c[occurances,ib].toarray())

        def sym_normalization_factor(key, quotient_func):
            assert self._csr is not None
            
            c = self._csr
            
            ii = m._label_index[key]

            fw_occurances = c[ii,:].nonzero()[1]
            bw_occurances = c[:,ii].nonzero()[0]

            return 1. / quotient_func(np.concatenate(
                [c[ii,fw_occurances].toarray().flatten(),
                 c[bw_occurances,ii].toarray().flatten()]
            ))
        
        def sym_p_a_given_b(key_a, key_b, quot_func = np.max):
            assert self._csr is not None
            
            c = self._csr
            
            ia = m._label_index[key_a]
            ib = m._label_index[key_b]

            v = c[ia,ib] + c[ib,ia]

            return v * self.sym_normalization_factor(key_b, quot_func)

        def fw_p_a_given_b(key_a, key_b, quot_func = np.max):
            assert self._csr is not None
            
            c = self._csr
            
            ia = m._x_label_index[key_a]
            ib = m._y_label_index[key_b]

            v = c[ia,ib]

            return v * self.bw_normalization_factor(key_b, quot_func)

        def bw_p_a_given_b(key_a, key_b, quot_func = np.max):
            assert self._csr is not None
            
            c = self._csr
            
            ia = m._y_label_index[key_a]
            ib = m._x_label_index[key_b]

            v = c[ib,ia]

            return v * self.fw_normalization_factor(key_b, quot_func)

        
        if self._sym:
            self.get_adjacent = get_sym_adjacent
            self.get_sum = sym_sum
            self.get_sym_normalization_factor = sym_normalization_factor
            self.p_a_given_b = sym_p_a_given_b
        
        else:
            self.get_forward_adjacent = get_forward_adjacent
            self.get_backward_adjacent = get_backward_adjacent
            
            self.get_fw_sum = fw_sum
            self.get_bw_sum = bw_sum
            
            self.get_fw_normalization_factor = fw_normalization_factor
            self.get_bw_normalization_factor = bw_normalization_factor

            self.fw_p_a_given_b = fw_p_a_given_b
            self.bw_p_a_given_b = bw_p_a_given_b





#!/usr/bin/env python3
import json
from abc import ABC, abstractmethod
class Model(ABC):

    def __init__(self, config):
        self.config = config


    # kernel sizes
    @abstractmethod
    def I_kernel_size(self):
        pass

    def A_kernel_size(self):
        pass

    def B_kernel_size(self):
        pass

    # matrices
    @abstractmethod
    def I(self, weights):
        pass

    @abstractmethod
    def A(self, weights):
        pass

    @abstractmethod
    def B(self, weights):
        pass



    @abstractmethod
    def get_number_of_states():
        pass

    @abstractmethod
    def get_number_of_emissions():
        pass

    @abstractmethod
    def state_id_to_str():
        pass

    @abstractmethod
    def str_to_state_id():
        pass

    @abstractmethod
    def emission_id_to_str():
        pass

    @abstractmethod
    def str_to_emission_id():
        pass

    def write_model(self):
        pass
    def read_model(self):
        pass
    def find_indices_of_zeros():
        pass

#!/usr/bin/env python3
import json
from abc import ABC, abstractmethod
class Model(ABC):

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def I():
        pass

    @abstractmethod
    def A():
        pass

    @abstractmethod
    def B():
        pass

    @abstractmethod
    def number_of_states():
        pass

    def write_model(self):
        pass

    def find_indices_of_zeros():
        pass

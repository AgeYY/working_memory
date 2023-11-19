import os.path as ospath
from sys import path as syspath
import matplotlib.pyplot as plt

syspath.insert(0, ospath.abspath(ospath.join(ospath.dirname(__file__), '..')))

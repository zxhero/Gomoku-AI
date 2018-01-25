## AlphaZero-Gomoku
This is an implementation of the AlphaZero algorithm for playing the simple board game Gomoku (also called Gobang or Five in a Row) from pure self-play training.
The game Gomoku is much simpler than Go or chess, so that we can focus on the training scheme of AlphaZero and obtain a pretty good AI model on a single PC in a few hours.
The AI can run on Pisvork.
The AI obey the restricted move rules in Gomoku.

References:  
1. AlphaZero: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
2. AlphaGo Zero: Mastering the game of Go without human knowledge


### Example Games Between Trained Models
- Each move  with 400 playouts:  
![playout400](https://raw.githubusercontent.com/junxiaosong/AlphaZero_Gomoku/master/playout400.gif)
- Each move  with 800 playouts:  
![playout800](https://raw.githubusercontent.com/junxiaosong/AlphaZero_Gomoku/master/playout800.gif)
>>>>>>> f45588fa03bf2af268deb06aeedffc22d9b96de7

### Requirements
To play with the trained AI models, only need:
- Python >= 2.7
- Numpy >= 1.11

To train the AI model from scratch, further need, either:
- Theano >= 0.7 and Lasagne >= 0.1      
or
- PyTorch >= 0.2.0

**PS**: if your Theano's version > 0.7, please follow this [issue](https://github.com/aigamedev/scikit-neuralnetwork/issues/235) to install Lasagne,  
otherwise, force pip to downgrade Theano to 0.7 ``pip install --upgrade theano==0.7.0``

If you would like to train the model using other DL frameworks, such as TensorFlow or MXNet, you only need to rewrite policy_value_net.py.

### Getting Started
To play with provided models, run the following script from the directory:  
```
python human_play.py  
```
You may modify human_play.py to try different provided models or the pure MCTS.

To train the AI model from scratch, with Theano and Lasagne, directly run:   
```
python train.py
```
With PyTorch, first modify the file [train.py](https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/train.py), i.e., comment the line
```
from policy_value_net import PolicyValueNet  # Theano and Lasagne
```
and uncomment the line 
```
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
```
and then execute: ``python train.py``  (To use GPU training, set ``use_gpu=True``)

The models (best_policy.model and current_policy.model) will be saved every a few updates (default 50).

**Tips for training:**
1. It is good to start with a 6 * 6 board and 4 in a row. For this case, we may obtain a reasonably good model within 500~1000 self-play games in about 2 hours.
2. For the case of 8 * 8 board and 5 in a row, it may need 2000~3000 self-play games to get a good model, and it may take about 2 days on a single PC.
3. For the case of 15 * 15 board and 5 in a row, it may need

I train it on win10 python3.4.0. You also need
    - libpython (https://www.lfd.uci.edu/~gohlke/pythonlibs/)
    - wheel (to install libpython)
    - mingw64
    - Theano >= 1.0
    - replace theano\compat\__init__.py line 41 with:
            def decode(x):
                return x.decode("GBK")

            def decode_iter(itr):
                for x in itr:
                    yield x.decode("GBK")
    - replace lasagne\layers\pool.py line 6 with:
            import theano.tensor.signal.pool as downsample

I also change some code in order to run it under python3.x
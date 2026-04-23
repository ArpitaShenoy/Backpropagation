# 🧠 MorphoGrad: Autograd from First Principles

A minimal, modular automatic differentiation engine built to bridge the gap between Mathematical Logic and Computational execution.

# ⚛️ The Philosophy

Most developers treat backpropagation as a "black box" library call. MorphoGrad treats it as a Natural Flow. By implementing the Chain Rule from scratch using NumPy, this project explores how error signals "flow" backward through a computational graph, much like energy distributing through a physical system.

# 🚀 Current State: V0 (The Foundation)

Engine: Support for forward and backward passes on basic Tensors.

Validation: All gradients are verified against PyTorch to ensure 100% mathematical parity.

Current implementation: Functional Linear layers with integrated affine transformations (xW+b).

# 🛠️ The "Leapfrog" Roadmap

I am not just building a library; I am mapping the evolution of Deep Learning architecture:

Refactor (Next Up): Transitioning to Operator Overloading. Moving math out of classes and into the Tensor core (__add__, __matmul__) to allow for natural y=x@w+b syntax.

Vision Phase: Implementing im2col based CNNs and ResNets from scratch.

Attention Phase: Scaling to Multi-head Attention and Vision Transformers (ViTs).

The Mission: Implementing Physics-Informed Neural Networks (PINNs) to bridge the gap between Puranas, Science, and AI.

# 🔬 Technical Highlight: The Chain Rule

Unlike high-level frameworks, MorphoGrad explicitly manages the Directed Acyclic Graph (DAG) of operations, ensuring that every leaf node receives its exact "contribution" to the final loss.
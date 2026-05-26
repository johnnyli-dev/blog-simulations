# blog-simulations

Code for figures and numerical experiments in posts on [my personal site](https://jonathanli.xyz/my-thoughts/adaptive-control-rk-tcp)

Each subdirectory corresponds to a single post and contains a self-contained script that produces the figure embedded in that post.

## Layout

- `adaptive-control-rk-tcp/` — RK45 adaptive step size and Jacobson/Karels RTO. Post: _Adaptive Step Sizes and TCP Retransmission Timers_.

## Running

```
pip install -r requirements.txt
cd <subdir>
python simulation.py
```

Each script writes `figure.png` into its own directory.

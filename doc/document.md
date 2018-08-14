# 海狮投资多因子模型技术文档

## 风险模型

### 收益归因

记投资组合权重向量为$\vec{w}=(w_1,\dots,w_N)^T$，投资组合的收益率为：
$$
\begin{array}{ll}
R_p & = \sum_nw_nr_n = \sum_{k=1}^Kf_k\sum_{n=1}^Nw_nX_{kn} + \sum_nw_n\mu_n\\
\\
    & = \sum_{k=1}^K\Psi_kf_k + \sum_nw_n\mu_n
\end{array}
$$
其中，$\Psi_k = \sum_{n=1}^Nw_nX_{kn}, \Psi = (w'X)'$，。由上式，可以将组合收益分解为各因子的风险收益和和特质风险的和，因此投资组合p对于因子k的风险暴露为$\Psi_k$，因子k对于投资组合收益的中贡献为$\Psi_kf_k$。

### 风险归因

记投资组合权重向量为$\vec{w}=(w_1,\dots,w_N)^T$，投资组合的波动率为：
$$
\begin{array}{ll}
\sigma_p & = \sqrt{\vec{w}'\Sigma\vec{w}}=\sqrt{(w'X)F(X'w) + w'\Lambda w}\\
\\
         & = \sqrt{\Psi'F\Psi + w'\Lambda w} = \sqrt{\sum_{k,l}\Psi_k\Psi_lF_{kl} + w'\Lambda w}
\end{array}
$$
其中，$\Psi_k = \sum_{n=1}^Nw_nX_{kn}, \Psi = (w'X)'$。参考Edward, Ronald & Erih中边际风险贡献(marginal contribution to risk, MCR)以及风险贡献(risk contribution, RC)的定义，即
$$
MCR_k = \frac{\partial\sigma_p}{\partial\Psi_k}, RC_k = \Psi_k\cdot\frac{\partial\sigma_p}{\partial\Psi_k} = \Psi_k\cdot MCR_k
$$
可验证，
$$
\begin{array}{ll}
RC_{total}^f & = \sum_{k=1}^KRC_k = \sum_{k=1}^K\Psi_k\frac{\partial\sigma_p}{\partial\Psi_k}\\
\\
             & = \frac{1}{\sigma_p}\sum_{k,l}\Psi_k\Psi_lF_{kl}\lt\sigma_p\\
\\
RC_k         & = \Psi_k\cdot\frac{\partial\sigma_p}{\partial\Psi_k} = \Psi_k\cdot\frac{1}{2\sigma_p}\cdot[2\sum_l\Psi_lF_{kl}]\\
\\
             & = \frac{1}{\sigma_p}\cdot\sum_l\Psi_k\Psi_lF_{kl}\\
\\
\vec{RC}     & = \Psi\cdot(F\cdot\Psi)
\end{array}
$$

## 收益模型


## 组合优化
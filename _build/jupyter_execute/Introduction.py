#!/usr/bin/env python
# coding: utf-8

# # Numerics for potential mean field games with hard constraints
# 
# We are in a potential case, that is to say the set of solutions to the mean field game system is linked with the set of solutions to a primal and dual potential problems. We define $\mathcal{T} = \{0,\ldots,T-1\}$, $\bar{\mathcal{T}} = \{0,\ldots,T\}$ and $S=\{0,\ldots,n\}$.
# For any $(t,s,x) \in \mathcal{T} \times \bar{\mathcal{T}}\times S$, the underlying mean field game system is given by
# 
# $$\begin{equation} \begin{cases} \begin{array}{cl} \text{(i)} &
# \begin{cases}
# \begin{array}{rl}
# u(t,x) = & \inf_{\rho \in \Delta(S)} \ell(t,x,\rho)\Delta_t + \gamma(t,x)\Delta_t + \sum_{y \in S} \rho(y)(\alpha(t,x,y)P(t)\Delta_t + u(t+1,y)), \\
# u(T,x) = & \gamma(T,x),
# \end{array}
# \end{cases}
# \\[2em]
# \text{(ii)} & \quad \pi(t,x, \cdot) \in \ \underset{\rho  \in \Delta(S)}{\text{arg min}}\ \ell(t,x,\rho)\Delta_t + \sum_{y \in S} \rho(y)(P(t) \alpha(t,x,y) \Delta_t + u(t+1,y)),\\[1.5em]
# \text{(iii)} &
# \begin{cases}
# \begin{array}{rl}
# m(t+1,x)= & \sum_{y \in S} m(t,y) \pi(t,y,x), \\
# m(0,x)= & m_0(x),
# \end{array}
# \end{cases}
# \\[2em]
# \text{(iv)} & \quad  \  {\displaystyle \gamma(s) \in \partial F (s,m(s)) }, \\[1.5em]
# \text{(v)} & \quad  \  {\displaystyle P(t) \in \partial \phi \Big(t,\sum_{(x,y) \in S^2}\pi(t,x,y) m(t,x) \alpha(t,x,y) \Big) },
# \end{array}
# \end{cases}
# \end{equation}
# $$

# ## Data
# 
# Here we provide the data of the problem we choose to solve.

# * __Running cost__:
# 
# $$\ell(t,x,\rho) = \sum_{y\in S} \rho(t,x,y)\beta(t,x,y), \quad \ell^\star(t,x,b) =  \max_{y\in S} b(t,x,y) - \beta(t,x,y), \quad \beta(t,x,y) = \left((y-x) \frac{\Delta_x}{\Delta_t}\right)^2/4,$$
# 
# where $\beta$ is a displacement cost
# 
# * __Congestion__ : 
# 
# $$F[m] = \|m\|^2/2 + \langle m, \nu \rangle + \chi_{[0,\eta]}(m),$$ 
# 
# * __Price__ : 
# 
# $$ \phi[D] = \frac{D_0}{2}\|D+\bar{D}\|^2 + \chi_{[D_{min},D_{max}]}(D),$$
# 
# * __weights__ $\alpha$:
# 
# $$\alpha(t,x,y) =  (y-x) \frac{\Delta_x}{\Delta_t},$$
# 
# for any $(t,x,y) \in \mathcal{T} \times S \times S$.

# ## Scaling
# In this subsection we explicit how the problem is scale in the numerical exemples. This section can be skipped in a first time. We set $\Delta_t = 1/T$ and $\Delta_x = 1/n$. We define a scalar product associated to each variable. For the primal variables we define
# 
# $$\begin{align} 
# \langle m_1, m_2 \rangle_{m} &= \sum_{t = 0}^{T-1} \Delta_x \Delta_t \langle m_1(t), m_2(t) \rangle + \Delta_x \langle m_1(T), m_2(T) \rangle, \\
# \langle w_1, w_2 \rangle_{w} &= \Delta_x \Delta_t \langle w_1, w_2 \rangle.
# \end{align}$$
# 
# For the dual variables we define
# 
# $$\begin{align}
# \langle u_1, u_2 \rangle_{u}  &= \Delta_x \langle u_1(0), u_2(0) \rangle + \sum_{t = 1}^{T} \Delta_x \Delta_t \langle u_1(t), u_2(t) \rangle, \\
# \langle \gamma_1, \gamma_2 \rangle_{\gamma}  &= \langle \gamma_1, \gamma_2 \rangle_{m},\\
# \langle P_1, P_2 \rangle_{P}  &= \Delta_t \langle P_1, P_2 \rangle.
# \end{align}$$ 

# ## Scaled primal and dual problems
# 
# ### Notations
# We have the following qualification result. For any function $f$, we define $f^\star$ its fenchel transform and $f^{\circ}$ its fenchel transform for the normalized scalar product. We use the same notation to denote the adjoint computed with the classical scalar product and the normalized scalar product.
# 
# We denote
# 
# $$ A^\star[P](t,x,y) = \alpha(t,x,y)P(t), \qquad S^\star[u](t,x,y) = u(t+1,y).$$
# 
# and 
# 
# $$
# \mathcal{A} = \begin{pmatrix} 0 & 0 & A\Delta_x & -id\\
# - I_0 - I/\Delta_t - I_T/\Delta_t & 0 & S/\Delta_t & 0\\
# id & -id & 0 & 0 \end{pmatrix}.$$
# 
# The adjoint operator of $\mathcal{A}$ is given by
# 
# $$
# \mathcal{A}^\circ  = \begin{pmatrix} 0 & - ( I_0 + I/\Delta_t + I_T/\Delta_t)^\circ & id\\
# 0 & 0 & -id \\
# A^\circ \Delta_x & S^\circ/\Delta_t  & 0 \\
# -id & 0 & 0 \end{pmatrix} = \begin{pmatrix} 0 & -(I_0/\Delta_t + I/\Delta_t + I_T) & id\\
# 0 & 0 & -id \\
# A^\star & S^\star/\Delta_t & 0 \\
# -id & 0 & 0\end{pmatrix}.$$
# 
# where
# 
# $$I_0[m] = \begin{cases} m(0) & \text{if } t=0\\
# 0 & \text{otherwise } \end{cases}, \quad I[m](t) = \begin{cases} m(t) & \text{if } 0<t<T\\
#  0 & \text{otherwise } \end{cases}, \quad I_T[m] = \begin{cases} m(T) & \text{if } t=T \\
# 0 & \text{otherwise } \end{cases}.$$
# 
# ### Duality
# 
# Under qualification assumption we have the following duality result
# 
# ```{admonition} Qualification
# 
# $$ \min_{(m_1,m_2,w,d) \in \mathcal{C}}
# \mathcal{F}(m_1,m_2,w,D) + \mathcal{G}(\mathcal{A}(m_1,m_2,w,D)) = \max_{(P,u,\gamma) \in \mathcal{K}} -\mathcal{F}^\circ(-\mathcal{A}^\circ(P,u,\gamma)) -  \mathcal{G}^\circ(P,u,\gamma).$$
# ```
# 
# where
# 
# $$ \begin{align} \mathcal{F}(m_1,m_2,w,D)  & = \sum_{(t,x) \in \mathcal{T} \times S} \tilde{\ell}[m_1,w](t,x) \Delta_t \Delta_x + \sum_{t \in \mathcal{T}} \phi[D](t)\Delta_t + F[m_2](t)\Delta_t \Delta_x + F[m_2](T)\Delta_x,\\
# \mathcal{G}(y_1,y_2,y_3)  & = \chi(y_1)  + \chi(y_2 + \bar{m}_0) + \chi(y_3). \end{align}$$
# 
# and
# 
# $$ \begin{align}  \mathcal{F}^\circ(a_1,a_2,b,c)  & = \sum_{t \in \bar{\mathcal{T}}} \tilde{\ell}^\star[a_1,b](t,x) \Delta_t \Delta_x +  \sum_{t \in \mathcal{T}} \phi^\star[c](t)\Delta_t + F^\star[a_2](t)\Delta_t \Delta_x + F^\star[a_2](T)\Delta_x,\\
# \mathcal{G}^\circ(P,u,\gamma)  & = - \langle u(0,\cdot), m_0(\cdot)\rangle,\end{align}$$
# 
# 

# ## Errors
# 
# For any $(t,s,x) \in \mathcal{T} \times \bar{\mathcal{T}}\times S$ we define $(\varepsilon_m, \varepsilon_\gamma, \varepsilon_P)$
# 
# ```{admonition} Errors
# 
# $$\begin{cases} 
# \varepsilon_\pi(t,x) & =  (\ell[\pi] +  \ell^\star[-A^\star P - S^\star u])(t,x) - \langle \pi(t,x),(A^\star P + S^\star u)(t,x) \rangle,\\[.5em]
#  \varepsilon_m(s,x)& =  G[m,\pi](s,x) - m(s,x) - \bar{m}_0(s,x),\\[.5em]
#  \varepsilon_\gamma(s) & = m(s) - \text{proj}_{\partial F^\star[\gamma](s)}(m(s)),
# \\[.5em]
# \varepsilon_P(t) & =   Q[m,\pi](t) - \text{proj}_{ \partial \phi^\star[P]}(Q[m,\pi](t)).
# \end{cases}$$
# ```
# 
# 
# 
# In our case we have that
# 
# $$\begin{align}
# F^\star[\gamma] &= \begin{cases} 0 & \text{ if } \gamma < 0,  \\ 
# \|\gamma\|^2 & \text{ if } \gamma \in [0,\eta], \\
# \eta & \text{ if } \gamma >\eta,
# \end{cases}, \\[1em] 
# \phi^\star[P] & = \begin{cases} \langle D_{min} ,P \rangle - D_0(D_{min} + \bar{D})^2/2 & \text{ if } P<D_0(D_{min} + \bar{D}), \\
# \|P\|^2/(2D_0) - \langle \bar{D} ,P \rangle & \text{ if } D_0(D_{min} + \bar{D}) \leq P \leq D_0(D_{max} + \bar{D}),\\
# \langle D_{max} ,P \rangle - D_0(D_{max} + \bar{D})^2/2 & \text{ if } D_0(D_{max} + \bar{D}) \leq P.
# \end{cases}
# \end{align}$$
# 
# Thus
# 
# $$\begin{align}
# \partial F^\star[\gamma] & = \nabla  F^\star[\gamma] = \begin{cases} 0 & \text{ if } \gamma < 0,  \\
# \gamma & \text{ if } \gamma \in [0,\eta], \\
# \eta & \text{ if } \gamma > \eta, 
# \end{cases}, \\[1em] \partial \phi^\star[P] & = \nabla \phi^\star[P] = \begin{cases} D_{min} & \text{ if } P<D_0(D_{min} + \bar{D}), \\ 
# P/D_0 - \bar{D} & \text{ if } D_0(D_{min} + \bar{D}) \leq P \leq D_0(D_{max} + \bar{D}),\\
#  D_{max} & \text{ if } D_0(D_{max} + \bar{D}) \leq P.
# \end{cases}
# \end{align}
# $$
# 
# and one can write
# 
# $$\begin{align}
#  \varepsilon_\gamma(s) & = m(s) - \nabla  F^\star[\gamma],
# \\
# \varepsilon_P(t) & = Q[m,\pi](t) - \nabla \phi^\star[P].
# \end{align} $$

# 
# ```{toctree}
# :hidden:
# :titlesonly:
# :caption: Algorithm for MFGC
# 
# Chambolle-Pock-MFGC-Constraint
# Chambolle-Pock-Bregman-MFGC-Constraint
# ADMM-MFGC-Constraint
# ADMG-MFGC-Constraint
# ```
# 
# 
# ```{toctree}
# :hidden:
# :titlesonly:
# :caption: Gallery
# 
# Convergence-Constraint
# Plots-MFG-Constraint
# Plots-MFGC-Constraint
# ```
# 

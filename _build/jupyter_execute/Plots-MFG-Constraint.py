#!/usr/bin/env python
# coding: utf-8

# # Plot MFG 

# ## Errors
# 
# |||
# |:---:|:---:|
# |<figure><img src="_images/chambolle_pock_mfg_constraint_error.png" width="400" async/> <figcaption>Chambolle-Pock</figcaption></figure>|<figure>  <img src="_images/chambolle_pock_bregman_constraint_mfg_error.png" width="400" async/> <figcaption>Chambolle-Pock-Bregman</figcaption></figure>|
# |<figure><img src="_images/relaxed_chambolle_pock_mfg_constraint_error.png" width="400" async/> <figcaption>Relaxed Chambolle-Pock</figcaption></figure>|<figure>  <img src="_images/inertial_chambolle_pock_mfg_constraint_error.png" width="400" async/> <figcaption>Inertial Chambolle-Pock</figcaption></figure>|
# |<figure> <img src="_images/ADMM_constraint_mfg_error.png" width="400"  async/><figcaption>ADMM</figcaption></figure> | <figure> <img src="_images/ADMG_constraint_mfg_error.png" width="400" async/><figcaption>ADM-G</figcaption></figure>|

# ## 3d Plots : measure, value and mean strategy
# 
# || measure| value  | mean stratgy |
# |:---:|:-----------:|:-----------:|:-----------:|
# |__Chambolle-Pock__|      <img src="_images/chambolle_pock_mfg_constraint_m.png" async/>     |      <img src="_images/chambolle_pock_mfg_constraint_u.png" async/>    |   <img src="_images/chambolle_pock_mfg_constraint_mean_strategy.png" async/> |
# |__Chambolle-Pock-Bregman__|      <img src="_images/chambolle_pock_bregman_constraint_mfg_m.png" async/>     |      <img src="_images/chambolle_pock_bregman_constraint_mfg_u.png" async/>    |   <img src="_images/chambolle_pock_bregman_constraint_mfg_mean_strategy.png" async/> |
# |__ADMM__ |     <img src="_images/ADMM_constraint_mfg_m.png" async/>      |      <img src="_images/ADMM_constraint_mfg_u.png" async/>    |  <img src="_images/ADMM_constraint_mfg_mean_strategy.png" async/> |
# |__ADMG__ |     <img src="_images/ADMG_constraint_mfg_m.png" async/>      |      <img src="_images/ADMG_constraint_mfg_u.png" async/>    |  <img src="_images/ADMG_constraint_mfg_mean_strategy.png" async/> |

# ## Contour polts : measure, value and mean strategy
# 
# 
# || measure| value  | mean stratgy |
# |:---:|:-----------:|:-----------:|:-----------:|
# |__Chambolle-Pock__|      <img src="_images/chambolle_pock_mfg_constraint_m_contour.png" async/>     |      <img src="_images/chambolle_pock_mfg_constraint_u_contour.png" async/>    |   <img src="_images/chambolle_pock_mfg_constraint_mean_strategy_contour.png" async/> |
# |__Chambolle-Pock-Bregman__|      <img src="_images/chambolle_pock_bregman_constraint_mfg_m_contour.png" async/>     |      <img src="_images/chambolle_pock_bregman_constraint_mfg_u_contour.png" async/>    |   <img src="_images/chambolle_pock_bregman_constraint_mfg_mean_strategy_contour.png" async/> |
# |__ADMM__ |     <img src="_images/ADMM_constraint_mfg_m_contour.png" async/>      |      <img src="_images/ADMM_constraint_mfg_u_contour.png" async/>    |  <img src="_images/ADMM_constraint_mfg_mean_strategy_contour.png" async/> |
# |__ADMG__ |     <img src="_images/ADMG_constraint_mfg_m_contour.png" async/>      |      <img src="_images/ADMG_constraint_mfg_u_contour.png" async/>    |  <img src="_images/ADMG_constraint_mfg_mean_strategy_contour.png" async/> |

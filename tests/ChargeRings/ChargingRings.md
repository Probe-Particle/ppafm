

Currently the code calculates average occupancy of the sites. It does it by thermodynamic integration of all possible occupncies of the spin orbitals. I consider 3 sites each witch one relevant molecular orbital (i.e. 2 spin orbitals). 

#### Electron Configurations

Therefore I have 6 spin-orbitals. This gives me 2^6 = 64 possible electron configurations. This is not too much so I just evaluate them all exhastively.  I represent them simply as binary numbers with 1 if the ith spin-orbital is occupied by electron and 0 if it is not. Like this:

$$
\psi_0 = [00|00|00] \newline
\psi_1 = [10|00|00] \newline 
\psi_2 = [01|00|00] \newline 
\psi_3 = [11|00|00] \newline 
 ...  \newline
\psi_{63} = [11|11|11] \newline
$$

In further equations I willl use in $d_k$ and $p_k$ for occupancy of up and down spin-orbitals on site $k$ (I choose $pd$ just because visually they resemble up and down arrows), $p$ and $d$ are integers (0 or 1) not floating point numbers.

For each such configuration I calculate the energy as 

$$ E_i = \sum_k \left [ ( d_k + p_k ) \left [ \varepsilon_k - \mu  + V_{tk}({\vec r}_t - {\vec r}_k ) \right ] +  d_kp_k C_k + (d_k+d_k) \sum_l K_{kl} (p_l+d_l) \right ] $$

There $\varepsilon_k$ is the original eigen energy of the molecular orbital for insulated molecule, $\mu$ is fermi level, $V_{tk}$ is the electrostatic interaction of the orbital at site $k$ with the tip, $C_k$ is the on-site Coulomb interaction (non-zero only if the site is uccupied by two electrons, with up and down spin), and   $K_{kl}$ is the Coulomb interaction between the sites $k$ and $l$. Currently the parameters $C_k$ and $K_{kl}$ are free parameters (we can thing if we should estimate them from some calculation, like DFT, or rather fit them to match experimental observation). $V_{tk}$ is the only parameter which depends on the position of the tip ${\vec r}_t$. That is, each pixel in the calculated image differ by different imput value of ${\vec r}_t$ therefore differetn terms $V_{tk}({\vec r}_t - {\vec r}_k )$ for each site $k$.


I model $V_{tk}$ for any ${\vec r}_{tk}={\vec r}_t - {\vec r}_k$ by multipole expansion as 

$$
V_{tk}({\vec r}_{tk}) = K_e q_t \left [  
    \frac{1}{|{\vec r}_{tk}| }  
    + \sum_{m \in \{x,y,z\}} \frac{ P_m {\hat Y_1^m}({\vec r}) }{|{\vec r}_{tk}|^2 }
    + \sum_{m \in \{xx,yy,zz,yz,xz,xy\}} \frac{ Q_m {\hat Y_2^m}({\vec r}) }{|{\vec r}_{tk}|^3 }
\right ]$$

where $K_e$ is the Coulomb constant, $q_t$ is the charge of the tip, $P_{h,k}$ and $Q_{h,k}$ are the dipole and quadrupole moments of the orbital $k$, and ${\hat Y_l^m}(\hat r)$ are the spherical harmonics calculated from the unit vector $\hat r_{tk} = {\vec r}_{tk}/|{\vec r}_{tk}|$. However in our stytem the dipoles are zero due to symmetry of the molecules.

NOTE: in fact, currnetly I calculate the multipole in cartesian basis rather than spherical harmonics, but I should probably change it to avoid problems with linear degeneracy of cartesian basis for quadrupoles

NOTE: Currently the bias dependence is modeled by tip charge $q_t$, and we consider tip as point-charge. The dependnece of $q_t$ on bias $q_t(V_{BIAS})$ is a question. Maybe we can consider some more sophisticated model for the field of the tip, but I don't think it will change anything qualitatice (i.e. experimentally measured shape of $dI/dV$ images resp. the charigning rings) 

#### Average Occupancy

With energy of each configuration $i$ I can easily evaluate the average occupancy (i.e. charge) of each site $\langle q_k \rangle$ (resp. each spin-orbital $\langle d_k \rangle$ and $\langle p_k \rangle $) simply by thermodynamic integration (sumation) of the energy dependent probability $w_i$ over all possible states $\psi_i $. 

$$ \langle d_k \rangle = \sum_i w_i  \langle \psi_i | d_k \rangle / \sum_i w_i $$
$$ \langle p_k \rangle = \sum_i w_i  \langle \psi_i | p_k \rangle / \sum_i w_i $$
$$ \langle q_k \rangle =\langle d_k \rangle +\langle p_k \rangle $$

where projection $\langle \psi_i | d_k \rangle$ (resp. $\langle \psi_i | p_k \rangle$  ) is just discrete occupancy (0 or 1) of the given spin orbital in the configuration $\psi_i$.

The question is how should I properly evaluate contribution $w_i$ ? Instictively I started with Boltzmann distribution 

$$ w_i = e^{-E_i/k_B T} $$

However, I am thinking that perhaps there should be Fermi distribution function (?). On the other hand I'm normalizing it by the sum over all configurations, which I guess it is excatly what makes Boltzmann distribution into Fermi distribution. If I would use Fermi distribution for $w_i$ then all states below fermi level (no mater how deep) would have the same weight, which does not seem reasonable. (Sorry, I'm a bit brainstorming here, I should look into some literature).

#### Remarks

* I don't see any problem to add other terms (perhaps spin-depenent) to the expression for energy $E_i$. In the end we just sum 64 configurations for like 400x400 pixels which is super fast (takes <0.01s).
* We can also compute full 3D volume (z-dependence) almost interactively, no problem.
* Maybe I miss something but what I'm doing is something like full-configuraton interaction just in very limited space of our 6 spin-orbitals. I don't see how more many-body this can be as long as we consider just this small sub-space.
* What can be maybe more chalanging is if you want to incorporate some quantum interaction with the continuous electron reservoir in the substrate (e.g. expressed by desity of states, self energy, greens functions, etc. ). 




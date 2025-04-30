from dmff import Hamiltonian, NeighborList
from dmff.generators.classical import PeriodicTorsionGenerator
from dmff.api.paramset import ParamSet
from dmff.api.xmlio import XMLIO
import optax

class DihedralTrainer:
    def __init__(
            self,
            xml,
            dihedraldata,
            loss_fn,
            optimizer,
            epochs):
        self.hamiltonian = Hamiltonian(xml)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.dihedraldata = dihedraldata
        self.epochs = epochs
        self.best_loss = float('inf')

    def train(self, data):
        # Initialize the optimizer
        opt_state = self.optimizer.init(self.hamiltonian.parameters)

        for epoch in range(self.epochs):
            for i, dihed in enumerate(len(self.dihedraldata)):
                loss_tmp, grads_tmp = self.loss_fn(paramset_dihed,
                                                   paramset0,
                                                   jnp_positions_list[i],
                                                   jnp_box_list[i],
                                                   jnp_pairs_list[i],
                                                   dihedral_qm_list[i])
                if i == 0:
                    loss = loss_tmp
                    grads = grads_tmp
                else:
                    loss += loss_tmp
                    grads = self._add_grad(grads, grads_tmp)
                
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_params = paramset_dihed
                
            updates, opt_state = self.optimizer.update(grads, opt_state)
            paramset_dihed = optax.apply_updates(paramset_dihed, updates)
            torsion_gen.overwrite(paramset_dihed)
            if epoch % 100 == 0:
                print(f"epoch: {epoch}, loss: {loss}")
            if epoch % 10000 == 0:
                io = XMLIO()
                io.writeXML(f"loop-{epoch}.xml", ffinfo)
        
    def _add_grad(grad1:ParamSet, grad2:ParamSet):
        for k in grad1.parameters.keys():
            for t in grad1.parameters[k].keys():
                grad1.parameters[k][t] += grad2.parameters[k][t]
        return ParamSet(grad1.parameters, grad1.mask)
    

class ThermodynamicTrainer:
    def __init__(
            self,
            xml,
            loss_fn,
            optimizer,
            epochs):
        self.hamiltonian = Hamiltonian(xml)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.best_loss = float('inf')

    def train(self):
        # Initialize the optimizer
        opt_state = self.optimizer.init(self.hamiltonian.parameters)

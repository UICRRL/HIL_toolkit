import math
import os
import torch
from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement, qNoisyExpectedImprovement
from botorch.acquisition.analytic import ProbabilityOfImprovement
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan, Interval
from botorch.sampling import IIDNormalSampler
from botorch.optim import optimize_acqf

# local imports
from HIL.optimization.kernel import SE, Matern 

import numpy as np
import matplotlib.pyplot as plt

# utils
import logging
from typing import Any, Optional, Tuple, Dict

    

import warnings
warnings.filterwarnings("ignore")


class BayesianOptimization(object):
    """
    Bayesian Optimization class for HIL
    """
    def __init__(self, n_parms:int = 1, range: np.ndarray = np.array([0,1]), noise_range :np.ndarray = np.array([0.005, 10]), acq: str = "ei", maximization : bool = True, \
        Kernel: str = "SE", model_save_path : str = "", device : str = "cpu" , plot: bool = False, optimization_iter: int = 500 , kernel_parms: Dict = {}) -> None:
        """Bayesian optimization for HIL

        Args:
            n_parms (int, optional): Number of optimization parameters ( exoskeleton parameters). Defaults to 1.
            range (np.ndarray, optional): Range of the optimization parameters. Defaults to np.array([0,1]).
            noise_range (np.ndarray, optional): Range of noise contraints for optimization. Defaults to np.array([0.005, 10]).
            acq (str, optional): Selecting acquisition function, options are 'ei', 'pi'. Defaults to "ei".
            Kernel (str, optional): Selecting kernel for the GP, options are "SE", "Matern". Defaults to "SE".
            model_save_path (str, optional): Path the new optimization saving directory. Defaults to "".
            device (str, optional): which device to perform optimization, "gpu", "cuda" or "cpu". Defaults to "cpu".
            plot (bool, optional): options to plot the gp and acquisition points. Defaults to False.
        """
        # TODO have an options of sending in the kernel parameters.
        if Kernel == "SE":
            self.kernel = SE(n_parms)
            self.covar_module = self.kernel.get_covr_module()

        else:
            self.kernel = Matern(n_parms)
            self.covar_module = self.kernel.get_covr_module()
        
        self.n_parms = n_parms
        self.range = range.reshape(2,self.n_parms).astype(float)
        self.maximization = maximization
        
        if len(model_save_path):
            self.model_save_path = model_save_path
        else:
            # this is temp
            self.model_save_path = "tmp_data/"

        self.optimization_iter = optimization_iter

        # place holder for model
        self.model = None

        # place to store the parameters
        self.x = torch.tensor([])
        self.y = torch.tensor([])

        # device 
        self.device = device

        # plotting
        self.PLOT = plot

        # logging
        self.logger = logging.getLogger()

        # Noise constraints
        self._noise_constraints = noise_range 
        self.likelihood = GaussianLikelihood() #noise_constraint=Interval(self._noise_constraints[0], self._noise_constraints[1]))

        # number of sampling in the acquisition points
        self.N_POINTS = 200

        # acquisition function type
        self.acq_type = acq

        if self.n_parms == 2:
            self.fig = plt.figure(figsize = (12,10))
            self.ax = plt.axes(projection='3d')

    def _step(self) -> np.ndarray:
        """ Fit the model and identify the next parameter, also plots the model if plot is true

        Returns:
            np.ndarray: Next parameter to sampled
        """

        parameter, value = self._fit()
        new_parameter = parameter.detach().cpu().numpy()

        self.logger.info(f"Next parameter is {new_parameter}")

        self._save_model()

        if self.PLOT:
            if self.n_parms == 1:
                self._plot()
            elif self.n_parms == 2:
                self._plot2d()

        return new_parameter

    def _get_data_best(self) -> float:
        """Get the best value predicted by the model

        Returns:
            float: best value
        """
        
        range = np.arange(self.range[0,:], self.range[1,:], self.N_POINTS)
        range = torch.tensor(range)
        self.model.eval() #type: ignore
        output = self.model(range)     #type: ignore
        return torch.max(output).detach().numpy() #type: ignore
    
    def _training(self, model, likelihood,train_x,train_y):

        """
        Train the model using Adam Optimizer and gradient descent
        Log Marginal Likelihood is used as the cost function
        """
           
        parameter = list(model.parameters()) + list(likelihood.parameters())
        optimizer = torch.optim.Adam(parameter, lr=0.01) 
        mll= ExactMarginalLogLikelihood(likelihood, model).to(train_x)
        

        train_y=train_y.squeeze(-1)
        loss = -mll(model(train_x), train_y) #type: ignore
        self.logger.info("before training Loss: ", loss.item())
        for i in range(500):
            
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y) #type: ignore
            
            loss.backward()
            optimizer.step()
        self.logger.info("after training Loss: ", loss.item()) 

    def _fit(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Using the model and likelihood select the next data point to get next data points and acq value at that point

        Returns:
            Tuple[torch.tensor, torch.tensor]: next parmaeter, value at the point
        """
        # tradition method
        # mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        # fit_gpytorch_model(mll) # check I need to change anything
        # using manual gradient descent using adam optimizer.
        self._training(self.model, self.likelihood, self.x, self.y)


        if self.acq_type == "ei":
            acq = qNoisyExpectedImprovement(self.model, self.x, sampler=IIDNormalSampler(self.N_POINTS, seed = 1234)) #type: ignore
        else:
            # TODO add other acquisition functions
            best_f = self._get_data_best()
            acq = ProbabilityOfImprovement(self.model, best_f, sampler=IIDNormalSampler(self.N_POINTS, seed = 1234)) #type: ignore
            pass
        new_point, value  = optimize_acqf(
            acq_function = acq,
            bounds=torch.tensor(self.range).to(self.device),
            q = 1,
            num_restarts=1000,
            raw_samples=2000,
            options={},
        )
        return new_point, value

    # Temp function will be replaced is some way
    def _plot(self) -> None:
        plt.cla()
        x = self.x.detach().numpy()
        y = self.y.detach().numpy()
        plt.plot(x, y, 'r.', ms = 10)
        x_torch = torch.tensor(x).to(self.device)
        y_torch = torch.tensor(y).to(self.device)
        self.model.eval()  #type: ignore
        self.likelihood.eval()
        with torch.no_grad():
            x_length = np.linspace(self.range[0,0],self.range[1,0],100).reshape(-1, 1)
            # print(x_length,self.range)
            observed = self.likelihood(self.model(torch.tensor(x_length))) #type: ignore
            observed_mean = observed.mean.cpu().numpy() #type: ignore
            upper, lower = observed.confidence_region() #type: ignore
            # x_length = x_length.cpu().numpy()
            
        plt.plot(x_length.flatten(), observed_mean)
        plt.fill_between(x_length.flatten(), lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.2)
        plt.legend(['Observed Data', 'mean', 'Confidence'])
        plt.pause(0.01)

    # Temp function will be replaced is some way
    def _plot2d(self) -> None:
        model=self.model
        model.eval() #type: ignore
        likelihood=self.likelihood
        likelihood.eval()
        self.ax.clear()
        x = self.x.detach().numpy()
        y = self.y.detach().numpy()
        y_mean = y.mean()
        y_std = y.std()
        with torch.no_grad():
            test_x = torch.linspace(self.range[0,0],self.range[1,0], 51).to(self.device)
            test_y = torch.linspace(self.range[0,1],self.range[1,1],51).to(self.device)
            XX,YY=torch.meshgrid(test_x,test_y,indexing='xy')
            #print(test_x.shape)
            XXX=torch.cat((XX.reshape(-1,1),YY.reshape(-1,1)),dim=1).double()
            observed_pred = likelihood(model(XXX)) #type: ignore

            # # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region() #type: ignore
            model_mean = observed_pred.mean*y_std + y_mean
            lower=lower*y_std + y_mean
            upper=upper*y_std + y_mean
            ZZ=model_mean.view(51,51)
            
            if max is False:
                ZZ = -ZZ
                lower = -lower
                upper = -upper
                y = -self.y 

            self.ax.plot_surface(XX.cpu().numpy(), YY.cpu().numpy(), ZZ.cpu().numpy(), cmap = 'winter',alpha=0.9) #type: ignore
            #print(XXX[:,0].shape,XXX[:,1].shape,lower1.shape)
            self.ax.plot_trisurf(XXX[:,0].cpu().numpy(), XXX[:,1].cpu().numpy(), lower.view(-1).cpu().numpy(),  #type: ignore
                    linewidth = 0.2,
                    antialiased = True,color='gainsboro',alpha=0.4,edgecolor='gainsboro') 
            self.ax.plot_trisurf(XXX[:,0].cpu().numpy(), XXX[:,1].cpu().numpy(), upper.view(-1).cpu().numpy(), #type: ignore
                    linewidth = 0.2,
                    antialiased = True,color='gainsboro',alpha=0.4,edgecolor='gainsboro')
            # find the location of minimum value
            print(ZZ.cpu().numpy().shape, XX.cpu().numpy().shape, YY.cpu().numpy().shape)
            ZZ = ZZ.cpu().numpy()
            XX = XX.cpu().numpy()
            YY = YY.cpu().numpy()
            min_index = np.unravel_index(ZZ.argmin(), ZZ.shape)
            logging.info(f"Min value is: {ZZ[min_index]}")
            logging.info(f"Min location (Timing) is: {XX[min_index]}")
            logging.info(f"Min location (Torque) is: {YY[min_index]}")
            self.min_location = np.array([XX[min_index], YY[min_index]])
            self.min_value = ZZ[min_index]

            # finding max value
            max_index = np.unravel_index(ZZ.argmax(), ZZ.shape)
            logging.info(f"Max value is: {ZZ[max_index]}")
            logging.info(f"Max location (Timing) is: {XX[max_index]}")
            logging.info(f"Max location (Torque) is: {YY[max_index]}")
            self.max_location = np.array([XX[max_index], YY[max_index]])
            self.max_value = ZZ[max_index]
            self.ax.scatter(x[:,0],x[:,1],y,color='r',marker='o',s=100,alpha=1) #type: ignore
            self.ax.view_init(25, -135) #type: ignore
            self.ax.set_zlabel("Cost",fontsize=18,rotation=90) #type: ignore
            self.ax.set_xlabel("Timing",fontsize=16)
            self.ax.set_ylabel("Torque",fontsize=16)
            # plt.pause(0.01)

    def _save_model(self) -> None:
        """Save the model and data in the given path
        """
        save_iter_path = self.model_save_path + f'iter_{len(self.x)}'
        os.makedirs(save_iter_path, exist_ok=True)
        model_path = save_iter_path +'/model.pth'
        torch.save(self.model.state_dict(), model_path) #type: ignore
        data_save = save_iter_path + '/data.csv'
        x = self.x.detach().cpu().numpy()
        y = self.y.detach().cpu().numpy()
        full_data = np.hstack((x,y))
        np.savetxt(data_save, full_data)
        self.logger.info(f"model saved successfully at {save_iter_path}")

    def run(self, x: np.ndarray, y: np.ndarray, reload_hyper: bool  = False ) -> np.ndarray:
        """Run the optimization with input data points

        Args:
            x (NxM np.ndarray): Input parameters N -> n_parms, M -> iter
            y (Mx1): Cost function array
            reload_hyper (bool, optional): Reload the hyper parameter trained in the previous iter. Defaults to True.

        Returns:
            np.ndarray: parameter to sample next
        """

        
        assert len(x) == len(y), "Length should be equal."

        self.x = torch.tensor(x).to(self.device)
        self.y = torch.tensor(y).to(self.device)

        if not reload_hyper:
            self.kernel.reset()
            self.likelihood = GaussianLikelihood(noise_constraint = Interval(self._noise_constraints[0], self._noise_constraints[1]))
            self.model = SingleTaskGP(self.x, self.y, likelihood = self.likelihood, covar_module = self.kernel.get_covr_module()) 
            # TODO check if this ok for multi dimension models
            self.model.to(self.device)

        else:
            # keeping the likehood save and kernel parameters so no need to reset those
            self.model = SingleTaskGP(self.x, self.y, likelihood = self.likelihood, covar_module = self.kernel.get_covr_module())
            self.model.to(self.device)

        # fi the model and get the next parameter.
        new_parameter = self._step()
        
        return new_parameter
        


if __name__ == "__main__":


    def mapRange(value, inMin=0, inMax=1, outMin=0.2, outMax=1.2):
        return outMin + (((value - inMin) / (inMax - inMin)) * (outMax - outMin))
    # objective function
    def objective(x, noise_std=0.0):
        # define range for input
        # r_min, r_max = 0, 1.2
        x = mapRange(x/100)
        return 	 -(1.4 - 3.0 * x) * np.sin(18.0 * x) + noise_std * np.random.random(len(x))
    # BO = BayesianOptimization(range = np.array([-0.5 * np.pi, 2 * np.pi]))
    BO = BayesianOptimization(range = np.array([0, 100]), plot=True)
    x = np.random.random(3) * 100

    y = objective(x)

    x = x.reshape(-1,1)
    y = y.reshape(-1, 1)
    
    full_x = np.linspace(0,100,100)
    full_y = objective(full_x)

    plt.plot(full_x, full_y)
    plt.plot(x,y, 'r*')
    plt.show()


    new_parameter = BO.run(x, y)
    # length_scale_store = []
    # output_scale_store = []
    # noise_scale_store = []
    for i in range(15):
        x = np.concatenate((x, new_parameter.reshape(1,1)))

        y = np.concatenate((y,objective(new_parameter).reshape(1,1)))
        new_parameter = BO.run(x,y)
        plt.pause(2)
    #     plt.show()
    #     # length_scale_store.append(BO.model.covar_module.base_kernel.lengthscale.detach().numpy().flatten())
    #     # output_scale_store.append(BO.model.covar_module.outputscale.detach().numpy().flatten())
    #     # noise_scale_store.append(BO.likelihood.noise.detach().numpy().flatten()
    #     # )
    # # x = np.linspace(-0.5 * np.pi, 2 * np.pi, 100)
    # x = np.linspace(0, 100, 100)
    # print('here')
    # plt.figure()
    # y = (np.sin(x/100)**3 + np.cos(x/100)**3) * 100

    # plt.plot(x,y,'r')
    # plt.figure()
    # plt.plot(length_scale_store, label='length scale')
    # plt.plot(output_scale_store, label = "sigma")
    # plt.plot(noise_scale_store, label="noise")
    # plt.legend()
    # plt.show()

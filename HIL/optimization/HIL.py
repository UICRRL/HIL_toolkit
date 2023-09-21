import numpy as np
import time
import pylsl
from typing import List


from HIL.optimization.BO import BayesianOptimization
from HIL.optimization.extract_cost import ExtractCost


class HIL:
    """Main HIL optimization
    This program will extract cost from the pylsl stream.
    Run the optimization.
    Check if the optimization is done.
    """

    def __init__(self, args: dict) -> None:
        """cost_name: name of the cost function."""
        self.n = int(0)  # number of optimization
        self.x = np.array([])  # input parameter for the exoskeleton
        # self.y = np.array([]) # cost function
        self.args = args

        # start the
        self.start_time = 0

        self._outlet_cost()

        self._reset_data_collection()

        # start optimization
        self._start_optimization(self.args["Optimization"])

        # start cost function
        self._start_cost(self.args["Cost"])

        # self.warm_up
        self.warm_up = True

        # start optimization
        self.OPTIMIZATION = False

        # normalization
        if self.args["Optimization"]["normalize"]:
            self.NORMALIZATION = True

        # The ones which are done.
        self.x_opt = np.array([])
        self.y_opt = np.array([])


    def _normalize_x(self, x: np.ndarray) -> np.ndarray:
        """Normalize x based on the range of the parameter

        Args:
            x (np.ndarray): x - parameters ( optimization parameters )

        Returns:
            np.ndarray: Normalized X parameters ( optimization parameters )
        """
        x = np.array(x).reshape(self.n, self.args["n_parms"])
        range_x = np.array(self.args["Optimization"]["range"]).reshape(2, self.args['n_parms'])
        x = (x - range_x[0,:]) / (range_x[1, :] - range_x[0, :])
        return x

    def _denormalize_x(self, x: np.ndarray) -> np.ndarray:
        """Denormalize x based on the range of the parameter, This is to send the parameters to exokseleton.

        Args:
            x (np.ndarray): x - parameters ( optimization parameters )

        Returns:
            np.ndarray: Denormalized X parameters ( optimization parameters )
        """
        x = np.array(x).reshape(self.n, self.args["n_parms"])
        range_x = np.array(self.args["Optimization"]["range"]).reshape(2, self.args['n_parms'])
        x = x * (range_x[1, : ] - range_x[0, :]) + range_x[0, :]
        return x

    def _mean_normalize_y(self, y: np.ndarray) -> np.ndarray:
        """Mean normalize y based on the range of the parameter

        Args:
            y (np.ndarray): y - cost function

        Returns:
            np.ndarray: Normalized y - cost function
        """
        y = np.array(y)
        y = (y - np.mean(y)) / np.std(y)
        return y

    def _outlet_cost(self) -> None:
        """Create an outlet function to send when the optimization has changed"""
        info = pylsl.StreamInfo(
            name="Change_parm",
            type="Marker",
            channel_count=self.args["n_parms"] + 1,
            source_id="12345",
        )
        self.outlet = pylsl.StreamOutlet(info)

    def _reset_data_collection(self) -> None:
        """Reset data collection and restart the clocks"""
        self.store_cost_data = []
        self.cost_time = 0
        self.start_time = 0

    def _start_optimization(self, args: dict) -> None:
        """O Start the optimization function, this will start the BO module

        Args:
            args (dict): Optimization args
        """
        print(args["range"][0], args["range"][1])
        print(np.array(list(args["range"])))
        self.BO = BayesianOptimization(
            n_parms=args["n_parms"],
            range=np.array(list(args["range"])),
            model_save_path=args["model_save_path"],
        )

    def _start_cost(self, args: dict) -> None:
        """Start the cost extraction module

        Args:
            args (dict): Cost args
        """
        self.cost_time = 0
        self.cost = ExtractCost(
            cost_name=args["Name"], number_samples=args["n_samples"]
        )

    def start(self):
        if self.n == 0:
            print(f"############################################################")
            print(f"############## Starting the optimization ###################")
            print(f"############## Using cost function {self.cost.cost_name} ###")
            print(f"############################################################")
            self._generate_initial_parameters()
            self.outlet.push_sample([0] * self.args["n_parms"] + [0])
        # start the optimization loop.
        while self.n < self.args["Optimization"]["n_steps"]:
            # Still in exploration
            if self.n < self.args["Optimization"]["n_exploration"]:
                print(
                    f"In the exploration step {self.n}, parameter {self.x[self.n]}, len_cost {len(self.store_cost_data)}"
                )

                if self.n == 0 and self.warm_up:
                    input(f"Please give 2 min of warmup and hit any key to continue \n")
                    self.warm_up = False

                self._get_cost()
                if (self.cost_time - self.start_time) > self.args["Cost"][
                    "time"
                ] and len(
                    self.store_cost_data
                ) > 5:  # 30 for 120
                    print(f" cost is {np.mean(self.store_cost_data[-5:])}")
                    out = input("Press Y to record the data: N to remove it:")
                    if out == "N":
                        self._reset_data_collection()
                        print("#########################")
                        print("########### recollecting #######")
                        print("#########################")
                    else:
                        if len(self.x_opt) < 1:
                            self.x_opt = np.array([self.x[self.n]])
                        else:
                            self.x_opt = np.concatenate(
                                (self.x_opt, np.array([self.x[self.n]]))
                            )
                        mean_cost = np.mean(self.store_cost_data[-5:])

                        if len(self.y_opt) < 1:
                            self.y_opt = np.array([mean_cost])
                        else:
                            self.y_opt = np.concatenate(
                                (self.y_opt, np.array([mean_cost]))
                            )

                        print(
                            f"recording cost function {self.y_opt[-1]}, for the parameter {self.x_opt[-1]}"
                        )

                        self.outlet.push_sample(
                            self.x_opt[-1].tolist() + [self.y_opt[-1]]
                        )
                        self._reset_data_collection()
                        self.n += 1
                        input("Enter to Continue")

            # Exploration is done and starting the optimization
            elif (
                self.n == self.args["Optimization"]["n_exploration"]
                and not self.OPTIMIZATION
            ):
                print(f" cost is {np.mean(self.store_cost_data[-5:])}")
                out = input("Press Y to record the data: N to remove it:")
                if out == "N":
                    self._reset_data_collection()
                    print("################################")
                    print("########### recollecting #######")
                    print("################################")
                else:
                    print(f"starting the optimization.")
                    if self.NORMALIZATION:
                        norm_x = self._normalize_x(self.x_opt)
                        norm_y = self._mean_normalize_y(self.y_opt)
                        new_parameter = self.BO.run(norm_x.reshape(self.n, -1), norm_y.reshape(self.n, -1))
                        new_parameter = self._denormalize_x(new_parameter)

                    else:
                        new_parameter = self.BO.run(
                            self.x_opt.reshape(self.n, -1),
                            self.y_opt.reshape(self.n, -1),
                        )
                    
                    print(f"Next parameter is {new_parameter}")
                    self.outlet.push_sample(self.x_opt[-1].tolist() + [self.y_opt[-1]])

                    # TODO Need to save the parameters and data for each iteration,
                    self.x = np.concatenate(
                        (
                            self.x,
                            new_parameter.reshape(
                                1,
                            ),
                        ),
                        axis=0,
                    )
                    self.OPTIMIZATION = True

            else:
                print(f"In the optimization loop {self.n}, parameter {self.x[self.n]}")
                self._get_cost()
                if (self.cost_time - self.start_time) > self.args["Cost"]["time"]:
                    out = input("Press Y to record the data: N to remove it:")
                    if out == "N":
                        self._reset_data_collection()
                        print("################################")
                        print("########### recollecting #######")
                        print("################################")
                    else:
                        self.x_opt = np.concatenate(
                            (self.x_opt, np.array([self.x[self.n]]))
                        )
                        mean_cost = np.mean(self.store_cost_data[-5:])
                        self.y_opt = np.concatenate((self.y_opt, np.array([mean_cost])))
                        self.n += 1
                        print(
                            f"recording cost function {self.y_opt[-1]}, for the parameter {self.x_opt[-1]}"
                        )
                        if self.NORMALIZATION:
                            norm_x = self._normalize_x(self.x_opt)
                            norm_y = self._mean_normalize_y(self.y_opt)
                            new_parameter = self.BO.run(norm_x.reshape(self.n, -1), norm_y.reshape(self.n, -1))
                            new_parameter = self._denormalize_x(new_parameter)

                        else:
                            new_parameter = self.BO.run(
                                self.x_opt.reshape(self.n, -1),
                                self.y_opt.reshape(self.n, -1),
                            )
                        print(f"Next parameter is {new_parameter}")
                        # TODO Need to save the parameters and data for each iteration
                        self.x = np.concatenate(
                            (
                                self.x,
                                new_parameter.reshape(
                                    1,
                                ),
                            ),
                            axis=0,
                        )
                        self.outlet.push_sample([self.x_opt[-1], self.y_opt[-1]])
                        self._reset_data_collection()
                        input("Enter to contiue")

            time.sleep(1)

    def _generate_initial_parameters(self) -> None:
        opt_args = self.args["Optimization"]
        self.x = (
            np.random.random(opt_args["n_start_points"])
            * (opt_args["range"][1] - opt_args["range"][0])
            + opt_args["range"][0]
        )
        # self.x[0]=35.0
        # self.x[1]=75.0
        # self.x[2]=10.0
        print(f"###### start functions are {self.x} ######")

    def _get_cost(self) -> None:
        """This function extracts cost from pylsl, need to be called all the time."""

        data, time_stamp = self.cost.extract_data()

        if time_stamp is not None:
            # changing maximization to minimization.
            data = data[-1] * -1
            self.cost_time = time_stamp
            self.store_cost_data.append(data)
            if len(self.store_cost_data) == 1:
                self.start_time = time_stamp
            print(
                f"got cost {self.store_cost_data[-1]}, parameter {self.x[self.n]}, time: {self.cost_time - self.start_time}"
            )

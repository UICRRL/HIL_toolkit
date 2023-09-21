import pylsl
import numpy as np
from typing import Any, Tuple

class ExtractCost:
    """
    Check for the streaming has cost, if so extract and updated the cost function.
    """

    def __init__(self, cost_name: str = 'Met_cost', number_samples: int = 2) -> None:
        self.stream = pylsl.resolve_streams()
        self.cost_name = cost_name
        # check the if the can cost is streaming
        COST_PRESENT = self._check_cost()
        if not COST_PRESENT:
            raise NameError
        self._setup_stream(number_samples)

        # self._setup_cost_stream(max_duration)
        self.data = np.array([])

    def _setup_stream(self, number_data: int = 2) -> None | pylsl.StreamInlet:
        self.inlet = None
        for info in self.stream:
            print(info)
            if info.name() == self.cost_name:
                self.inlet = pylsl.StreamInlet(info, max_buflen=number_data)
        return self.inlet
        
        

    def _check_cost(self) -> bool:
        """
        Check if the cost is streaming
        :return:
        """
        for info  in self.stream:
            if info.name() == self.cost_name:
                print(f"Found {info.name} in the stream")
                return True
            
            # if not specific cost function is found search for any cost function.
            elif "cost" in info.name():
                print(f"NOT FOUND {self.cost_name} in the stream")
                print(f"Found {info.name} in the stream change the cost name to that to continue")
            else:
                print(f"NOT Cost stream found")
        
        return False

    def extract_data(self, t:int = 30) -> Tuple[Any, Any]:
        """
        Extract data from inlet.
        """
        # self.inlet.time_correction()
        # y = np.empty((1024))
        if self.inlet is not None:
            y,time_stamp = self.inlet.pull_sample(timeout=0.01)
            if y is not None:
                print(len(y))
                self.inlet.flush()
            return y, time_stamp
        else:
            print("Check the stream and restart this code......")
            return None, None
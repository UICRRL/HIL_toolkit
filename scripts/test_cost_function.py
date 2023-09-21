

from HIL.cost_processing.utils.inlet import InletOutlet
import pylsl
import numpy as np
import time

def cost_function(x):
    x = np.array(x)
    return sum(- (x - 5 ) ** 2  + 100 + np.random.rand(1) * 0.1)


class test_cost_function:
    """Test the cost function class"""
    # create a fake stream
    def __init__(self) -> None:
        info = pylsl.StreamInfo(name='test_function', type='cost', channel_count=1,
                                nominal_srate=pylsl.IRREGULAR_RATE,
                                channel_format=pylsl.cf_double64,
                                source_id='test_function_id')
        self.outlet = pylsl.StreamOutlet(info)
        pylsl.local_clock()
        self.inlet = None

        self._get_streams()
        
        self.x_parmeter = [1, 1]
        self.parameters = np.empty((100, 2))

    def _get_streams(self) -> None:
        """Get the streams from the inlet"""
        print("looking for an Change_parm stream...")
        streams = pylsl.resolve_byprop("name", "Change_parm", timeout=1)
        if len(streams) == 0:
           pass
        else:
            self.inlet = pylsl.StreamInlet(streams[0])


    def _get_cost(self) -> None:
        """Get the cost function"""
        sample, timestamp = self.inlet.pull_sample()

        print(sample, timestamp)
        if timestamp:
            print(timestamp, sample)
            self.inlet.flush()
            pass
            # sample = sample[-1]
            # print(f"Received {sample} at time {timestamp}")
            # self.x_parmeter = sample[:2]

    def run(self,) -> None:
        """Main run function for the cost function this will take the parameters and send the cost function to the outlet.

        Args:
            x_parmeter (float): parameter for the cost function.
        """
        counter = 0
        while True:
            time.sleep(0.1)
            print("running")
            if self.inlet is None:
                self._get_streams()
            else:
                self._get_cost()
                x_parmeter = self.x_parmeter
                # print(x_parmeter, cost_function(x_parmeter))
                self.outlet.push_sample([cost_function(x_parmeter)])
                # if counter > 10:
                #     del self.inlet
                #     self._get_streams()




if __name__ == "__main__":
    test = test_cost_function()
    test.run()
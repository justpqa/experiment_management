from threading import Thread

class CustomThread(Thread):
    def __init__(self, target=None, args=(), kwargs={}, stop_event = None):
        super().__init__()
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.stop_event = stop_event
        
    def run(self):
        while self.stop_event is not None and not self.stop_event.is_set():
            if self.target is not None:
                try:
                    self.target(*self.args, **self.kwargs)
                    print("The process has been completed")
                    return
                except:
                    # only run when stop event is triggered, the thread will have 1 iteration of error before the stop_event is triggered
                    print("The process has been stopped")
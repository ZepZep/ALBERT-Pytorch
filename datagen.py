import multiprocessing as mp
import setproctitle

class DataWorker(mp.Process):
    def __init__(self, idx, config, inqueue, outqueue, *args, **kwargs):
        super(mp.Process, self).__init__(*args, **kwargs)
        self.idx = idx
        self.name = f"python zep DataWorker {idx}"
        self.config = config
        self.context = None
        self.inqueue = inqueue
        self.outqueue = outqueue
        self.should_run = True

    def run(self):
        try:
            setproctitle.setproctitle(self.name)
            self.context = self._init_context()

            while self.should_run:
                try:
                    par = self.inqueue.get(block=True, timeout=1)
                except mp.TimeoutError:
                    continue

                example = self._create_example(par)
                self.outqueue.put(example)
        except KeyboardInterrupt:
            print(f"Worker {self.idx} interrupted!\n", flush=True, end="")

        self._destroy_context(self.context)

    def _create_example(self, par):
        pass

    def _init_context(self):
        pass

    def _destroy_context(self, context):
        pass


class DataFeeder(mp.Process):
    def __init__(self, idx, config, queue, *args, **kwargs):
        super(mp.Process, self).__init__(*args, **kwargs)
        self.config = config
        self.queue = queue
        self.idx = idx
        self.name = f"python zep DataFeeder {idx}"

    def run(self):
        setproctitle.setproctitle(self.name)
        self.feed()
        
    def feed(self):
        pass


class DataGen:
    def __init__(self, config, worker_count,
                 worker_class=DataWorker, feeder_class=DataFeeder):
        self.config = config
        self.worker_count = worker_count
        self.worker_class = worker_class
        self.feeder_class = feeder_class
        
    def __call__(self):
        return self

    def __iter__(self):
        return DataIter(self.config, self.worker_count,
                        self.worker_class, self.feeder_class)


class DataIter:
    id_counter = 0

    def __init__(self, config, worker_count,
                 worker_class=DataWorker, feeder_class=DataFeeder):
        self.config = config
        self.worker_count = worker_count
        self.worker_class = worker_class
        self.feeder_class = feeder_class

        self.idx = self.id_counter
        self.id_counter += 1
        
        #print(f"Creating DataGenIter {self.id_counter}")

        self.inqueue = mp.Queue(worker_count)
        self.outqueue = mp.Queue(worker_count)

        self.feeder = feeder_class(f"{self.idx}:F", config, self.inqueue)
        self.workers = [
            worker_class(f"{self.idx}:{i}", config, self.inqueue, self.outqueue)
                for i in range(worker_count)
        ]

        self.feeder.start()
        for worker in self.workers:
            worker.start()

    def __next__(self):
        return self.outqueue.get()

    def __del__(self):
        print(f"Deleting DataGenIter {self.id_counter}")
        
        self.feeder.terminate()
        
        for worker in self.workers:
            worker.terminate()

        print(f"Finished deleting DataGenIter {self.id_counter}")
        #for worker in self.workers:
            #worker.join()



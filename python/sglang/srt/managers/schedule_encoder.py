from queue import Queue

class ScheduleEncoderMixin:
    def init_encoder(self):
        # TODO: 开关
        self.mm_encoder_waiting_queue = Queue()
        self.mm_encoder_end_queue = Queue()
        self._init_mm_encoder_worker()
    
    def _recv_req(self, recv_reqs):
        # TODO: 开关
        if True:
            while not self.mm_encoder_end_queue.empty():
                recv_req = self.mm_encoder_end_queue.get()
                recv_reqs.append(recv_req)
            
            while True:
                try:
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                    if (
                        isinstance(recv_req, TokenizedGenerateReqInput)
                        and recv_req.mm_inputs is not None
                    ):
                        self.mm_encoder_waiting_queue.put(recv_req)
                    else:
                        recv_reqs.append(recv_req)
                except zmq.ZMQError:
                    break

        else:
            while True:
                try:
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                recv_reqs.append(recv_req)
    
    def _init_mm_encoder_worker(self):
        pass
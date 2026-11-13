import threading
import unittest
from classical.native import NativeEngine


class NativeFenceTests(unittest.TestCase):
    def test_unconfirmed_idle_quarantines_engine_and_refuses_new_work(self):
        engine=NativeEngine.__new__(NativeEngine);engine.quarantined=False;engine.lock=threading.Lock()
        engine.control=lambda *a:[{'is_processing':True}]
        with self.assertRaises(RuntimeError):engine.wait_idle(seconds=.001)
        self.assertTrue(engine.quarantined)
        # Refusal happens before tokenization, socket creation or another generation.
        with self.assertRaises(RuntimeError):engine.generate_messages([])

if __name__=='__main__':unittest.main()

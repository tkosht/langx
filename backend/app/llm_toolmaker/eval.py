import re
from threading import BoundedSemaphore, Lock, Thread

from omegaconf import DictConfig
from tqdm import tqdm
from typing_extensions import Self

from app.base.component.logger import Logger
from app.llm_toolmaker.bbh import get_option, get_task, get_wrapper, is_option_selection
from app.llm_toolmaker.tool_user import LlmToolUser

g_logger = Logger(logger_name="app")


# cf. https://github.com/ctlllll/LLM-ToolMaker.git
class LlmToolEvaluator(object):
    def __init__(
        self,
        wrapper: str,
        task_name: str = "example_task",
        model_name: str = "gpt-3.5-turbo",
        max_threads: int = 8,
    ) -> None:
        self.wrapper: str = wrapper
        self.task_name: str = task_name
        self.model_name: str = model_name

        self.max_threads: int = max_threads
        self.pool = BoundedSemaphore(self.max_threads)
        self.lock = Lock()

        self.tool_user = LlmToolUser(
            wrapper=wrapper, model_name=model_name, task_name=task_name
        )

        self.n_totals: int = 0
        self.n_corrects: int = 0

    def _adjust(self, ans: str, sample):
        if is_option_selection(ans, sample):
            options = (
                re.findall(r"Options:(.*)", sample["question"], re.DOTALL)[0]
                .strip()
                .split("\n")
            )
            for option in options:
                if ans in option:
                    ans = option.split(" ")[0]
                    break

        if self.task_name == "schedule_meeting":
            if ans is None:
                ans = "No time slot works."
            elif isinstance(ans, list) or isinstance(ans, tuple):
                ans = f"{ans[0]} - {ans[1]}"

        ans = get_option(ans)
        return ans

    def run(self, sample: dict):
        with self.pool:
            try:
                ans = self.tool_user.make_answer_from_sample(
                    task=self.task_name, sample=sample
                )
                ans = self._adjust(ans, sample)
            except Exception as e:
                ans = f"Error: {e}"
            with self.lock:
                self.n_totals += 1
                if str(ans) == str(sample["answer"]):
                    self.n_corrects += 1
                else:
                    g_logger.info(f"incorrect: {ans=} / {sample['answer']=}")
                acc = self.n_corrects / self.n_totals
                g_logger.info(f"Thread Accuracy: {acc:.4f}")

    def eval(self, testset: list) -> Self:
        threads = []
        for sample in tqdm(testset, desc="creating threads"):
            thr = Thread(target=self.run, args=(sample,))
            threads.append(thr)
            thr.start()
            # self.run(sample)  # for debugging
            # break

        thr_bar = tqdm(threads, desc="waiting threads: ")
        for thr in thr_bar:
            thr.join()

        acc = self.n_corrects / self.n_totals
        g_logger.info(f"Last Accuracy: {acc:.4f}")
        return self


def _main(params: DictConfig):
    task_name = params.task_name
    trainset, validsdet, testset = get_task(task=task_name)
    wrapper = get_wrapper(task=task_name, tooldir="./llm_tools")

    lte = LlmToolEvaluator(task_name=task_name, wrapper=wrapper)
    lte.eval(testset=testset[:50])


# @from_config(params_file="conf/app.yml", root_key="/train")
# def config(cfg: DictConfig):
def config():
    cfg = DictConfig(dict(task_name="word_sorting"))
    return cfg


def main(
    task_name: str = None,
):
    s = signature(main)
    kwargs = {}
    for k in list(s.parameters):
        v = locals()[k]
        if v is not None:
            kwargs[k] = v

    params = config()  # use as default
    params.update(kwargs)
    return _main(params)


if __name__ == "__main__":
    from inspect import signature

    import typer

    typer.run(main)

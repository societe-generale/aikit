import pandas as pd
from aikit import enums
from aikit.tools.helper_functions import diff, intersect


def unnest_job_parameters(job):
    """ unnest a param dictionnary,
    skipping 'ColumnsSelector' and 'columns_to_use'
    """
    res = {
        'job_id': job["job_id"]
    }
    for k, v in job["all_models_params"].items():
        if k[1][1] != "ColumnsSelector":
            for k2, v2 in v.items():
                if k2 != "columns_to_use":
                    res["__".join(k[1]) + "__" + k2] = v2

            res[k[1][0]] = k[1][1]

    for b in job["blocks_to_use"]:
        res["hasblock_%s" % b] = 1

    return res


class ResultsReader:

    def __init__(self, storage):
        self.storage = storage

        self.jobs = {}
        self.results = {}

    def load(self):
        self._load_cache(self.jobs, 'jobs')
        self._load_cache(self.results, 'finished')

    def _load_cache(self, cache, folder):
        files = self.storage.listdir(folder)
        for file in files:
            file_name = file.split('.')[0]
            if file_name not in cache:
                if folder == 'jobs':
                    cache[file_name] = self.storage.load_special_json(file_name, folder)
                else:
                    cache[file_name] = self.storage.load_json(file_name, folder)

    def get_jobs_params(self) -> pd.DataFrame:
        self._load_cache(self.jobs, 'jobs')
        params = [unnest_job_parameters(job) for job in self.jobs.values()]
        params = pd.DataFrame(params)

        # Fill Missing Value for step columns
        for cat in enums.StepCategories.alls:
            if cat in list(params.columns):
                params.loc[params[cat].isnull(), cat] = "--nothing--"

        # Fill Missing Value for block use
        for c in params.columns:
            if c.startswith("hasblock_"):
                params.loc[params[c].isnull(), c] = 0

        # Re-order columns
        def reorder_param_col(cols):
            return sorted(cols, key=lambda x: tuple(x.split("__")))

        cols = list(params.columns)
        cols_hasblock = sorted([c for c in cols if c.startswith("hasblock_")])

        cols = intersect(["job_id"], cols) + cols_hasblock + reorder_param_col(diff(cols, ["job_id"] + cols_hasblock))
        assert sorted(cols) == sorted(list(params.columns))
        params = params.loc[:, cols]
        return params

    def get_results(self, agregate=False) -> pd.DataFrame:
        self._load_cache(self.results, 'finished')
        results = [pd.DataFrame(r['results']).assign(job_id=job_id)
                   for job_id, r in self.results.items()
                   if 'results' in r]
        if len(results) == 0:
            return pd.DataFrame(columns=['job_id'])
        results = pd.concat(results)
        if agregate:
            results = results.groupby('job_id').mean().reset_index()
        return results

    def get_additional_results(self) -> pd.DataFrame:
        self._load_cache(self.results, 'finished')
        results = [pd.DataFrame(r['additional_results']).assign(job_id=job_id)
                   for job_id, r in self.results.items()
                   if 'additional_results' in r]
        if len(results) == 0:
            return pd.DataFrame(columns=['job_id'])
        results = pd.concat(results)
        return results

    def get_number_of_successful_jobs(self) -> int:
        self._load_cache(self.results, 'finished')
        results = [r for _, r in self.results.items() if 'results' in r]
        return len(results)

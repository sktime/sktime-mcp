from pathlib import Path

p = Path('src/sktime_mcp/runtime/executor.py')
content = p.read_text()

new_method = """    def predict_interval(self, handle_id, fh=None, coverage=0.9):
        \"\"\"Generate probabilistic prediction intervals.\"\"\"
        try:
            instance = self._handle_manager.get_instance(handle_id)
        except KeyError:
            return {'success': False, 'error': f'Handle not found: {handle_id}'}
        if not self._handle_manager.is_fitted(handle_id):
            return {'success': False, 'error': 'Estimator not fitted'}
        try:
            if fh is None:
                fh = list(range(1, 13))
            intervals = instance.predict_interval(fh=fh, coverage=coverage)
            result = {}
            if isinstance(intervals, pd.DataFrame):
                intervals.index = intervals.index.astype(str)
                for idx in intervals.index:
                    row = intervals.loc[idx]
                    result[str(idx)] = {'lower': float(row.iloc[0]), 'upper': float(row.iloc[-1])}
            return {'success': True, 'intervals': result, 'coverage': coverage, 'horizon': len(fh)}
        except Exception as e:
            return {'success': False, 'error': str(e)}

"""

content = content.replace('    def fit_predict(', new_method + '    def fit_predict(')
p.write_text(content)
print('Done')
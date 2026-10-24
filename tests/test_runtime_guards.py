"""Exercise launcher failures without CUDA, pip, network, or training data."""
import ast
import json
import sys
import types
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def function(name, env):
    path = ROOT / 'src/train_seg.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
    return env[name]


def launcher_case(code, artifacts, gpu_count=1):
    out = Path(tempfile.mkdtemp(prefix='launcher_', dir=ROOT / 'scratch'))
    parser = types.SimpleNamespace(add_argument=lambda *a, **k: None,
                                   parse_args=lambda: types.SimpleNamespace(fold=None, out=str(out), config=None))
    def popen(*a, **kw):
        kw['stdout'].write('Synthetic process, no actual training\n')
        kw['stdout'].flush()
        for pattern in artifacts:
            (out / pattern.format(fold=0)).write_bytes(b'synthetic')
        return types.SimpleNamespace(returncode=code, poll=lambda: code)
    lines = []
    sys.modules['torch'] = types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda: gpu_count), __version__='synthetic')
    env = dict(argparse=types.SimpleNamespace(ArgumentParser=lambda: parser), Path=Path,
               CFG={'folds_to_run': [0]}, json=json, sys=sys, os=__import__('os'),
               time=types.SimpleNamespace(sleep=lambda _: None),
               subprocess=types.SimpleNamespace(run=lambda *a, **k: None, Popen=popen, STDOUT=-2),
               __file__=str(ROOT / 'src/train_seg.py'), print=lambda *a, **k: lines.append(a))
    failed = False
    try:
        function('main', env)()
    except RuntimeError:
        failed = True
    status = json.loads((out / 'RUN_STATUS.json').read_text()) if (out / 'RUN_STATUS.json').exists() else None
    return failed, status, ('ALL DONE',) in lines


def main():
    (ROOT / 'scratch').mkdir(exist_ok=True)
    results = {}
    failed, status, done = launcher_case(1, [])
    assert failed and status['status'] == 'FAILED' and not done
    results['failed_child_propagates'] = True
    failed, status, done = launcher_case(0, ['model_f{fold}.pt'])
    assert failed and status['failures'][0]['missing'] == ['probs_f0.npz', 'log_f0.json'] and not done
    results['missing_outputs_rejected'] = True
    failed, status, done = launcher_case(0, ['model_f{fold}.pt', 'probs_f{fold}.npz', 'log_f{fold}.json'])
    assert not failed and status['status'] == 'COMPLETED' and done
    results['complete_outputs_accepted'] = True
    failed, status, done = launcher_case(0, [], 0)
    assert failed and status is None and not done
    results['no_gpu_rejected_without_launch'] = True
    calls = []
    def unet(encoder, **kwargs):
        calls.append(encoder)
        raise OSError('synthetic pretrained download failure')
    sys.modules['segmentation_models_pytorch'] = types.SimpleNamespace(Unet=unet)
    try:
        function('build_model', {})('efficientnet-b4')
    except RuntimeError:
        pass
    else:
        raise AssertionError('Failed requested model must stop')
    assert calls == ['efficientnet-b4']
    results['encoder_failure_does_not_change_architecture'] = True
    (ROOT / 'scratch' / 'GUARD_TEST_RESULTS.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

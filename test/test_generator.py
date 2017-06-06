import os
import subprocess
import unittest


class TestGenerator(unittest.TestCase):
    def test(self):
        destination = 'test_template'
        template_dir = os.path.join('tensorkit', 'templates')
        subprocess.call(
            'tk-generate %s -v' % destination,
            shell=True
        )

        cwd = os.getcwd()
        destination = os.path.join(cwd, destination)
        
        src = os.path.join(template_dir, 'architect.py')
        dst = os.path.join(destination, 'architect.py')
        self.compare_file(src, dst)

        src = os.path.join(template_dir, 'dataset.py')
        dst = os.path.join(destination, 'dataset.py')
        self.compare_file(src, dst)

        src = os.path.join(template_dir, 'evaluator.py')
        dst = os.path.join(destination, 'evaluator.py')
        self.compare_file(src, dst)

        src = os.path.join(template_dir, 'objective.py')
        dst = os.path.join(destination, 'objective.py')
        self.compare_file(src, dst)

        src = os.path.join(template_dir, 'optimizer.py')
        dst = os.path.join(destination, 'optimizer.py')
        self.compare_file(src, dst)

        src = os.path.join(template_dir, 'hypes.json')
        dst = os.path.join(destination, 'hypes.json')
        self.compare_file(src, dst)

    def compare_file(self, src, dst):
        self.assertTrue(os.path.isfile(src), 'Source is not a file: ' + src)
        self.assertTrue(os.path.isfile(dst), 'Destination is not a file: ' + dst)

        with open(src) as f:
            src_content = f.read()

        with open(dst) as f:
            dst_content = f.read()

        self.assertTrue(src_content == dst_content, 'Content of source file and destination file are different.')
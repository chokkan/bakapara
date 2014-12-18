#!/usr/bin/env python

"""Bakapara client for IPython clusters.

Copyright (C) 2014  Naoaki Okazaki

Distributed under the terms of the BSD License.  The full license is in
the file COPYING, distributed as part of this software.

"""

import datetime
import os
import socket
import sys
import time
from IPython.core.display import clear_output
from IPython.parallel import Client, TimeoutError

def runjob(job):
    import os
    import subprocess
    import socket
    from IPython.parallel.error import UnmetDependency

    # Determine if this engine (host) can accept the job.
    if 'host' in job and socket.getfqdn() not in job['host']:
        # UnmetDependency exception forces the scheduler to resubmit the task to a different engine.
        raise UnmetDependency

    # Make sure that 'cmd' field exists in the job.
    if 'cmd' not in job:
        return dict(error='"cmd" field does not exist in the job')

    # Change the working directory if specified.
    if 'cd' in job:
        os.chdir(job['cd'])

    # Set stdout and stderr file objects if specified.
    stdout = open(job['out'], 'w') if 'out' in job else None
    stderr = open(job['err'], 'w') if 'err' in job else None

    # Run the command.
    try:
        # We force to use '/bin/bash -o pipefail' so that the return value presents the value of the rightmost command to exit with a non-zero status code.
        returncode = subprocess.call(job['cmd'], shell=True, executable='/bin/bash -o pipefail', stdout=stdout, stderr=stderr)
        return dict(code=returncode)
    except OSError, e:
        return dict(error=str(e))

class Bakapara:
    """A "bakapara" client for IPython cluster.

    Args: identical to IPython.parallel.Client

    """
    def __init__(self, **args):
        self.rc = Client(**args)
        self.lview = None
        self.ar = None
        self.jobs = None
        self.indices = None
        self.finished = set()
        # Obtain the host names and PIDs of engines.
        self.pids = self.rc[:].apply(os.getpid).get_dict()
        self.hosts = self.rc[:].apply(socket.getfqdn).get_dict()
        # Change the working directory of each engine.
        self.rc[:].apply(os.chdir, os.getcwd())

    def run(self, jobs, targets=None):
        """Runs the jobs on the cluster.

        Args:
            jobs (list): list of dictionary objects describing the jobs.
            targets (int, list of ints, 'all', or None): the engine(s) on which the jobs will run.

        Returns:
            bool: True if successful, False otherwise (e.g., jobs are running).

        """
        if self.ar is not None and not self.ar.ready():
            return False
        self.lview = self.rc.load_balanced_view(targets)
        self.ar = self.lview.map_async(runjob, jobs)
        self.jobs = jobs
        self.indices = dict([(k, v) for v, k in enumerate(self.ar.msg_ids)])
        self.finished = set()
        return True

    def wait(self, timeout=1e-3):
        """Waits for the jobs to complete and writes job results.

        Args:
            timeout (float): a time in seconds, after which to give up.

        """
        if self.ar is None:
            return
        
        # Find finished msg_ids.
        pending = set(self.ar.msg_ids)
        try:
            self.rc.wait(pending, timeout)
        except TimeoutError:
            pass
        finished = pending.difference(self.rc.outstanding)
        
        # Overwrite the results in the job array.
        for msg_id in finished:
            i = self.indices[msg_id]
            if i in self.finished:
                continue
            job = self.jobs[i]
            meta = self.rc.metadata[msg_id]
            result = self.rc.results[msg_id][0]
            for key in ('submitted', 'started', 'completed', 'received'):
                result[key] = meta[key].isoformat()
            for key in ('engine_id', 'pyerr', 'pyout', 'status', 'msg_id'):
                result[key] = meta[key]
            result['elapsed'] = str(meta['completed'] - meta['started'])
            result['host'] = self.hosts[meta['engine_id']]
            job['result'] = result
            self.finished.add(i)

    def ready():
        """Returns whether the jobs have completed."""
        return self.ar is not None or self.ar.ready()

    def successful():
        """Returns whether the jobs completed without raising an exception.

        Raises:
            AssertionError: the result is not ready.

        """
        return self.ar is not None and self.ar.successful()

    def abort(self, **args):
        """Aborts jobs.

        Args: identical to IPython.parallel.client.view.LoadBalancedView.abort()
        
        """
        if self.lview is not None:
            self.lview.abort()

    def interrupt(self):
        """Sends SIGINT signal to engines (experimental).

        http://mail.scipy.org/pipermail/ipython-dev/2014-March/013426.html
        """
        self.abort()
        for i in self.rc.ids:
            host = self.hosts[i]
            pid = self.pids[i]
            if host == socket.getfqdn():
                os.kill(pid, signal.SIGINT)
            else:
                os.system('ssh {} kill -INT {}'.format(host, pid))

    def shutdown(self, **args):
        """Terminates one or more engine processes, optionally including the hub.

        Args: identical to IPython.parallel.Client.shutdown

        """
        if self.lview is None:
            return False
        self.lview.shutdown(**args)
            
    def status(self, interval=1., timeout=-1, fo=sys.stdout):
        """Waits for the jobs, printing progress at regular intervals

        Args:
            interval (float): a time in seconds, after which to print the progress.
            timeout (float): a time in seconds, after which to give up waiting.
            fo (file): a file object to which the progress is printed.

        """
        if self.ar is None:
            return
        if timeout is None:
            timeout = -1

        # Make sure to write the job results into the job objects.
        self.wait(1e-3)

        tic = time.time()
        while not self.ar.ready() and (timeout < 0 or time.time() - tic <= timeout):
            self.wait(interval)
            clear_output(wait=True)
            dt = datetime.timedelta(seconds=self.ar.elapsed)
            fo.write('{}/{} tasks finished after {}'.format(self.ar.progress, len(self.ar), str(dt)))
            fo.flush()
        else:
            fo.write('\n')
        dt = datetime.timedelta(seconds=self.ar.elapsed)
        clear_output(wait=True)
        fo.write('{} tasks completed in {}\n'.format(len(self.ar), str(dt)))
        fo.flush()

    def __len__(self):
        """Returns the number of engines."""
        return len(self.rc)

def main(args=None):
    import argparse
    import json
    import logging as lg
    import operator

    # Parse the command-line arguments.
    parser = argparse.ArgumentParser(
        description="Run jobs on the IPython cluster environment in 'bakapara' manner."
        )
    parser.add_argument(
        '-p', '--profile', default='default',
        help='specify the name of the Cluster profile'
        )
    parser.add_argument(
        '-i', '--input',
        help='specify the filename for reading the jobs (default: STDIN)'
        )
    parser.add_argument(
        '-o', '--output',
        help='specify the filename for storing the results (default: STDOUT)'
        )
    parser.add_argument(
        '-l', '--logging',
        help='specify the filename for storing the logs (default: STDERR)'
        )
    args = parser.parse_args(args)

    # Determine the stream for reading the jobs.
    fi = open(args.input) if args.input else sys.stdin

    # Determine the stream for storing the results.
    fo = open(args.output, 'w') if args.output else sys.stdout

    # Determine the stream for storing the logs.
    fe = open(args.logging, 'w') if args.logging else sys.stderr

    # Initialize the default logger.
    lg.basicConfig(level=lg.INFO, format='%(asctime)s %(message)s', stream=fe)

    # Read the jobs.
    jobs = map(lambda line: json.loads(line), fi)

    # Initialize Bakapara client.
    bp = Bakapara(profile=args.profile)

    # Run the jobs.
    lg.info('total {} jobs on {} engines'.format(len(jobs), len(bp)))
    bp.run(jobs)

    # Loop until all jobs are done.
    finished = set()
    while len(finished) < len(jobs):
        bp.wait(1.)
        just_finished = bp.finished - finished
        if just_finished:
            finished_jobs = [jobs[i] for i in just_finished]
            finished_jobs.sort(key=lambda x: x['result']['completed'])
            i = len(finished) + 1
            for job in finished_jobs:
                fo.write('{}\n'.format(json.dumps(job)))
                fo.flush()
                lg.info('[{i}/{n}] returned {code} in {elapsed} sec on {host}: {cmd}'.format(i=i, n=len(jobs), code=job['result']['code'], elapsed=job['result']['elapsed'], host=job['result']['host'], cmd=job['cmd']))
                i += 1
            finished |= bp.finished

    lg.info('completed')
    fo.close()
    fe.close()

if __name__ == '__main__':
    main()

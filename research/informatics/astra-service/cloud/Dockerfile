FROM public.ecr.aws/lambda/python:3.12
RUN dnf upgrade -y && dnf install -y libgomp gmp mpfr && dnf clean all
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt cudaq==0.16.0 boto3
RUN python -c "import cudaq; cudaq.set_target('qpp-cpu'); k=cudaq.make_kernel(); q=k.qalloc(1); k.x(q[0]); assert cudaq.sample(k,shots_count=4)['1']==4"
COPY workspace /var/task/workspace
COPY runtime.py destinations.py /var/task/
ENV TMPDIR=/tmp OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 JOBLIB_MULTIPROCESSING=0
CMD ["runtime.handler"]

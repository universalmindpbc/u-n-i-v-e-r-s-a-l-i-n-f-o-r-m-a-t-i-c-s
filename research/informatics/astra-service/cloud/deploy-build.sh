#!/usr/bin/env bash
set -euo pipefail
export AWS_PAGER=""
region=us-east-1
stack=informatics-astra-staging-bootstrap
test "$(aws sts get-caller-identity --query Account --output text)" = "740168708062"
aws cloudformation validate-template --template-body file://bootstrap.template.json --region "$region" >/dev/null
aws cloudformation deploy --template-file bootstrap.template.json --stack-name "$stack" --capabilities CAPABILITY_IAM --region "$region" --no-fail-on-empty-changeset
aws cloudformation describe-stacks --stack-name "$stack" --region "$region" --query 'Stacks[0].Outputs' --output json > outputs.json
bucket=$(python3 -c 'import json; print(next(x["OutputValue"] for x in json.load(open("outputs.json")) if x["OutputKey"]=="Bucket"))')
aws s3 cp source.zip "s3://$bucket/build/source.zip" --region "$region" --only-show-errors
aws s3 cp dispatcher.zip "s3://$bucket/build/dispatcher.zip" --region "$region" --only-show-errors
aws codebuild start-build --project-name informatics-astra-staging --region "$region" --environment-variables-override "name=IMAGE_TAG,value=${1:?source revision required},type=PLAINTEXT" --query 'build.id' --output text | tee build-id.txt

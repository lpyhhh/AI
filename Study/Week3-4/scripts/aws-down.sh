#! /bin/bash
# 如何使用？
# 配合不同机型，更改位置
## 

<<EOF
删除和上传的时间尽量短-控制在3H内，会有sopt终端风险
EOF
# 1. 确保目标路径的所有文件被删除
aws s3 rm "$1" --recursive --only-show-errors

# 2. 立即执行同步上传
aws s3 sync "$2" "$1" --only-show-errors

# aws ec2 stop-instances --instance-ids i-07e995cc57ccb7091 --region us-east-1
# aws ec2 cancel-spot-instance-requests --spot-instance-request-ids sir-yyyyyyyyyyyyyyyyy --region your-region
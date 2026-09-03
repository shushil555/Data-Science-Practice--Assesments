import boto3

s3 = boto3.client('s3')

# Create bucket
bucket_name = 'prt661-life-expectancy'

# Upload raw data
s3.upload_file('data/life_expectancy.csv', bucket_name, 'raw/life_expectancy.csv')

# Upload processed data
s3.upload_file('data/processed_data.csv', bucket_name, 'processed/processed_data.csv')

print("Files uploaded to S3 successfully")
print(f"Raw data: s3://{bucket_name}/raw/life_expectancy.csv")
print(f"Processed data: s3://{bucket_name}/processed/processed_data.csv")

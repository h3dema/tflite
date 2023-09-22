
This keys are provided as an example.
PLEASE, don't use them in any production computer.
Since the private key is here that can be a breach of security.


## How to `re`create the keys

The following command generates the key pair again.
Just change the e-mail shown in the `-C` parameter.


```
ssh-keygen -C henrique.duartemoura@uantwerpen.be -f id_tflite
```

If you change the name of the file with the key, you also will need to change the name in the Dockerfile.
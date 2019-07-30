from subprocess import check_output
import subprocess



command="java --add-modules java.xml.bind -cp eval.jar Main.Aspects /sem8/COL772-NLP/DE-CNN/data/official_data/pred.xml /sem8/COL772-NLP/DE-CNN/data/official_data/Laptops_Test_Gold.xml"
print(check_output(command.split()))
# subprocess.run(command)
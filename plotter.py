import matplotlib.pyplot as plt
def pol():
   with open('stress.txt', "r") as f:
      int_list = list(map(float, (line.strip() for line in f)))
      print(int_list) 

      with open('time.txt', "r") as t:
         ti = list(map(str, (line.strip() for line in t)))
      print(ti) 


      
   plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
   plt.bar(ti, int_list, color='skyblue')

      # Add labels and title
   plt.xlabel("Time")
   plt.ylabel("Integer Value")
   plt.title("Bar Graph of Integer Values vs. Time")

      # Display the graph
   plt.xticks(ti)  # Set the x-axis ticks to the time values
   plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add a faint grid on the y-axis
   plt.tight_layout()
      #plt.show()
   plt.savefig('plot.png') 
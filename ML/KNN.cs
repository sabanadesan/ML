using System;
using ML.Yuk;

namespace ML
{
	public class KNN
	{
		public static void Test1()
        {
			Vector v1 = new Vector(1, 3);
			Vector v2 = new Vector(2, 5);

			Labelled data = new Labelled(new Labels(v1, "Flower"), new Labels(v2, "Plant"));

			string l = Model.KNNClassify(5, data, new Vector(4, 6));

			Console.WriteLine(l);
		}

		public static void Test2()
        {
            string path = GetPath("iris", "data");
            NDArray data = new NDArray(typeof(double), typeof(double), typeof(double), typeof(double), typeof(string));
            NDArray cols = new NDArray("sepal_length", "sepal_width", "petal_length", "petal_width", "class");

            DataFrame df = DataFrame.LoadCsv(path, ',', false, null, data, true);
        }

        public static String GetPath(String symbol, String csvDir)
        {
            // Append symbol name to file extension
            string fileName = symbol + ".csv";

            // Get current directory from build folder
            string currentDir = Directory.GetCurrentDirectory();

            // Append current directory to csv directory
            string path = Path.Combine(currentDir, csvDir);

            // Append csv directory path to file name
            var filePath = Path.Combine(path, fileName);

            return filePath;
        }
    }
}


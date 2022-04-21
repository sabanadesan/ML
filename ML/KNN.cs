using System;
using ML.Yuk;

namespace ML
{
	public class KNN
	{
		public static void Test1()
        {
			Vector v1 = new Vector(1, 3, 4);
			Vector v2 = new Vector(2, 5, 3);

			Labelled data = new Labelled(new Labels(v1, "Flower"), new Labels(v2, "Plant"));

			string l = Model.KNNClassify(5, data, new Vector(5, 5, 5));

			Console.WriteLine(l);
		}

		public static void Test2()
        {
            string path = GetPath("iris", "data");
            NDArray data = new NDArray(typeof(double), typeof(double), typeof(double), typeof(double), typeof(string));
            NDArray cols = new NDArray("sepal_length", "sepal_width", "petal_length", "petal_width", "class");

            DataFrame df = DataFrame.LoadCsv(path, ',', false, cols, data);

            NDArray nd = df.GetValue();

            NDArray xs = nd[new Slice(0, nd.Length - 1), new Slice(0, nd[0].Length)];
            NDArray ys = nd[new Slice(nd.Length - 1, nd.Length), new Slice(0, nd[0].Length)];

            var o = Model.TrainTestSplit(xs, ys, 0.7, true);

            NDArray x_train = o.XTrain;
            NDArray x_test = o.XTest;
            NDArray y_train = o.YTrain;
            NDArray y_test = o.YTest;

            Labelled l = new Labelled();

            for(int i = 0; i < x_train[0].Length; i++)
            {
                NDArray d = Model.GetRow(x_train, i);
                Vector v = new Vector(d.AsType<dynamic>());
                dynamic label = y_train[0, i];

                Labels labels = new Labels(v, label);

                l.Add(labels);
            }

            int w = 0;

            for (int i = 0; i < x_test[0].Length; i++)
            {
                NDArray d = Model.GetRow(x_test, i);
                Vector v = new Vector(d.AsType<dynamic>());
                dynamic label = y_test[0, i];

                string p = Model.KNNClassify(5, l, v);

                if (p.Equals(label))
                {
                    w++;
                }
            }

            int t = x_test[0].Length;

            System.Console.WriteLine((double) w/t);
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


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
	}
}


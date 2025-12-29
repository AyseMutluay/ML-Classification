using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ClassificationOdev
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "data.txt");

        static void Main(string[] args)
        {
            Console.WriteLine("=== ML.NET Siniflandirma Odevi Basliyor ===");

            // --- 1. EĞİTİM AŞAMASI ---
            MLContext mlContext = new MLContext();

            // Veriyi yükle
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, separatorChar: ',', hasHeader: false);

            Console.WriteLine("Veri seti yuklendi, model egitiliyor...");

            // Pipeline oluştur (Metni işle -> Algoritmayı seç)
            var pipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            // Modeli eğit
            ITransformer model = pipeline.Fit(dataView);

            Console.WriteLine("Model egitimi tamamlandi!");
            Console.WriteLine("===========================================");

            // --- 2. TEST AŞAMASI (YENİ EKLENEN KISIM) ---

            // Tahmin Motorunu Oluştur (Tekli tahminler için gereklidir)
            var predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            Console.WriteLine("\nTEST MODU AKTIF");
            Console.WriteLine("Bir cumle yazip Enter'a bas (Cikmak icin 'exit' yaz):");
            Console.WriteLine("-----------------------------------------------------");

            while (true)
            {
                Console.Write("Cumle Giriniz: ");
                string input = Console.ReadLine();

                // Çıkış kontrolü
                if (input == "exit" || string.IsNullOrWhiteSpace(input)) break;

                // Tahmin yap
                var prediction = predictionEngine.Predict(new SentimentData { SentimentText = input });

                // Sonucu ekrana yaz (True: Olumlu, False: Olumsuz)
                string sonuc = prediction.Prediction ? "OLUMLU :)" : "OLUMSUZ :(";

                // Probability: Modelin ne kadar emin olduğu (0 ile 1 arası)
                Console.WriteLine($"-> Sonuc: {sonuc} | Guven Orani: {prediction.Probability:P2}");
                Console.WriteLine("");
            }
        }
    }

    // --- SINIFLAR ---
    public class SentimentData
    {
        [LoadColumn(0)]
        public string SentimentText { get; set; }

        [LoadColumn(1), ColumnName("Label")]
        public bool Sentiment { get; set; }
    }

    public class SentimentPrediction : SentimentData
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
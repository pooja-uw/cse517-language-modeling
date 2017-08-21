import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.io.Source

object Common {

  /**
    * Defining constants for Unknown, start, stop, punctuation and numbers
    */

  val Unk: String = "<<UNK>>"
  val UnkThreshold: Int = 1
  val Start: String = "<<START>>"
  val Stop: String = "<<STOP>>"
  val Punc: String = "<<PUNC>>"
  val Num: String = "<<NUM>>"

  /**
    * This method loads training corpus and testing corpus from source and preprocess both corpus and create final training, testing and dev corpus.
    *
    * @param trainingCorpusName - Training corpus name
    * @param testingCorpusName  - Testing corpus name
    * @return This function returns normalized dev corpus, training corpus, testing corpus, dictionary, ngramcount and ngramprobabilities.
    */

  def loadCorpus(trainingCorpusName: String, testingCorpusName: String = null) = {

    val completeTrainingCorpus = Source.fromFile(s"$trainingCorpusName.txt", "ISO-8859-1").mkString.split("\n").toList
    val trainingCorpusChunks = splitCorpus(replaceNonWords(completeTrainingCorpus))
    val trainingChunk = trainingCorpusChunks(0)

    var testingCorpus: List[String] = null
    if (testingCorpusName == null) {
      testingCorpus = trainingCorpusChunks(1)
    }
    else {
      val completeTestingCorpus = Source.fromFile(s"$testingCorpusName.txt", "ISO-8859-1").mkString.split("\n").toList
      val testingCorpusChunks = splitCorpus(replaceNonWords(completeTestingCorpus))
      testingCorpus = testingCorpusChunks(1)
    }

    var trainingChunks: List[List[String]] = splitCorpus(trainingChunk)
    val trainingCorpus = trainingChunks(0)
    var devTrainingCorpus = trainingChunks(1)

    val Tuple2(dictionary: Map[String, Int], wordFrequency: Map[Int, Int]) = prepareDictionaryAndComputeFrequency(trainingCorpus)

    val unknowns = computeUnknowns(wordFrequency)

    val normalizedTrainingCorpus = normalizeTrainingCorpus(trainingCorpus, dictionary, unknowns)
    val nGramCounts = computeNGramCounts(normalizedTrainingCorpus, dictionary)
    val wordCount = nGramCounts._1.values.sum

    val normalizedDevTrainingCorpus = normalizeTestingCorpus(devTrainingCorpus, dictionary, unknowns)
    val normalizedTestingCorpus = normalizeTestingCorpus(testingCorpus, dictionary, unknowns)


    val nGramParameters = computeNGramParameters(dictionary, nGramCounts._1, nGramCounts._2, nGramCounts._3, normalizedTrainingCorpus.length, wordCount)

    Tuple6(normalizedTrainingCorpus, normalizedDevTrainingCorpus, normalizedTestingCorpus, dictionary, nGramCounts, nGramParameters)
  }

  /**
    * This function split corpus in 80:20 proportion
    *
    * @param corpus
    * @return
    */
  def splitCorpus(corpus: List[String]) = {
    val corpusLines = corpus
    // Random.shuffle(corpus)
    val trainingChunkSize = (corpusLines.length * 0.8).floor.asInstanceOf[Int]
    List(corpusLines.slice(0, trainingChunkSize), corpusLines.drop(trainingChunkSize))
  }

  /**
    * This function calculate bag of unknowns
    *
    * @param wordFrequency - Map of words and its frequency
    * @return Map of unknown words with its frequency
    */
  def computeUnknowns(wordFrequency: Map[Int, Int]): Map[Int, Int] = {
    wordFrequency.filter((word) => word._2 <= UnkThreshold)
  }

  /**
    * This function is used to make dictionary and word frequency map where each word is assigned unique id
    *
    * @param corpus - Training corpus
    * @return Returns two maps one for dictionary and other for word frequency
    */
  def prepareDictionaryAndComputeFrequency(corpus: List[String]): (Map[String, Int], Map[Int, Int]) = {
    val dictionary = mutable.Map[String, Int]()
    val wordFrequencies = mutable.Map[Int, Int]()

    dictionary(Common.Unk) = 0
    dictionary(Common.Start) = 1
    dictionary(Common.Stop) = 2
    dictionary(Common.Num) = 3
    dictionary(Common.Punc) = 4

    var wordIndex: Int = 5

    corpus.foreach((line: String) => {
      line.split("[\\s]+").foreach(word => {
        if (!dictionary.contains(word)) {
          dictionary(word) = wordIndex
          wordIndex += 1
        }
        wordFrequencies(dictionary(word)) = wordFrequencies.getOrElse(dictionary(word), 0) + 1
      })
    })

    Tuple2(dictionary.toMap, wordFrequencies.toMap)
  }

  /**
    * This method replaces punctuations and numbers with punctuationa nd number constant in corpus
    *
    * @param corpus - Corpus
    * @return Returns modified corpus
    */
  def replaceNonWords(corpus: List[String]) = {
    val modifiedCorpus = ListBuffer[String]()
    corpus.foreach(line =>
      modifiedCorpus += line
        .replaceAll("\\p{Punct}+", " " + Punc + " ")
        .replaceAll("\\p{Digit}+", " " + Num + " ")
    )
    modifiedCorpus.toList
  }

  /**
    * This method replaces words in training corpus with unique ids allocated to words
    *
    * @param trainingCorpus - Input corpus
    * @param dictionary     - Map of words to its unique id
    * @param unknowns       - Bag of unknowns
    * @return - Returns processed training corpus
    */

  def normalizeTrainingCorpus(trainingCorpus: List[String], dictionary: Map[String, Int], unknowns: Map[Int, Int]) = {
    var adjustedCorpus = ListBuffer[String]()
    trainingCorpus.foreach((line: String) => {
      adjustedCorpus += line.split("[\\s]+")
        .map((word: String) => if (unknowns.contains(dictionary(word))) dictionary(Unk) else dictionary(word))
        .mkString(" ")
    })
    adjustedCorpus.toList
  }

  /**
    * This method replaces words in testing with unique ids allocated to words in dictionary. If word is not in dictionary assign it to UNK
    *
    * @param trainingCorpus - Input testing corpus
    * @param dictionary     - Map of words to unique ids
    * @param unknowns       _ Bag of unknowns
    * @return Return processed testing corpus
    */
  def normalizeTestingCorpus(trainingCorpus: List[String], dictionary: Map[String, Int], unknowns: Map[Int, Int]) = {
    var adjustedCorpus = ListBuffer[String]()
    trainingCorpus.foreach((line: String) => {
      adjustedCorpus += line.split("[\\s]+")
        .map((word: String) => if (dictionary.contains(word) && unknowns.contains(dictionary(word))) dictionary(Unk) else dictionary.getOrElse(word, dictionary(Unk)))
        .mkString(" ")
    })
    adjustedCorpus.toList
  }

  /**
    * This function computes unigram, bigram and trigram count for training corpus
    *
    * @param trainingCorpus _ Input training corpus
    * @param dictionary     - Map of words to unique ids
    * @return Returns unigram, bigram and trigram counts
    */
  def computeNGramCounts(trainingCorpus: List[String], dictionary: Map[String, Int]) = {
    val unigramCounts = mutable.Map[Int, Int]()
    val bigramCounts = mutable.Map[Tuple2[Int, Int], Int]()
    val trigramCounts = mutable.Map[Tuple3[Int, Int, Int], Int]()

    trainingCorpus.foreach((line: String) => {
      val words = (dictionary(Start) + " " + dictionary(Start) + " " + line + " " + dictionary(Stop)).split("[\\s]+")
      words.zipWithIndex.foreach { case (word: String, position: Int) => {

        /**
          * Compute Unigram counts.
          */
        if (position > 1) {
          unigramCounts(word.toInt) = unigramCounts.getOrElse(word.toInt, 0) + 1
        }

        /**
          * Compute Bigram counts.
          */
        if (position > 0 && position < words.length - 1) {
          bigramCounts(Tuple2(word.toInt, words(position + 1).toInt)) = bigramCounts.getOrElse(Tuple2(word.toInt, words(position + 1).toInt), 0) + 1
        }

        /**
          * Compute Trigram counts.
          */
        if (position < words.length - 2) {
          trigramCounts(Tuple3(word.toInt, words(position + 1).toInt, words(position + 2).toInt)) = trigramCounts.getOrElse(Tuple3(word.toInt, words(position + 1).toInt, words(position + 2).toInt), 0) + 1
        }
      }
      }
    })

    Tuple3(unigramCounts.toMap, bigramCounts.toMap, trigramCounts.toMap)
  }

  /**
    *
    * @param dictionary
    * @param unigramCounts
    * @param bigramCounts
    * @param trigramCounts
    * @param lineCount
    * @param wordCount
    * @return
    */
  def computeNGramParameters(dictionary: Map[String, Int], unigramCounts: Map[Int, Int], bigramCounts: Map[Tuple2[Int, Int], Int], trigramCounts: Map[Tuple3[Int, Int, Int], Int], lineCount: Int, wordCount: Int) = {
    val unigramProbability = mutable.Map[Int, Double]()
    val bigramProbability = mutable.Map[Tuple2[Int, Int], Double]()
    val trigramProbability = mutable.Map[Tuple3[Int, Int, Int], Double]()
    val bigramHistoryProbablity = mutable.Map[Int, Double]()
    val bigramHistoryCount = mutable.Map[Int, Int]()
    val trigramHistorySet = mutable.Map[Tuple2[Int, Int], mutable.Set[Int]]()

    unigramCounts.foreach { case (word: Int, count: Int) => {
      unigramProbability(word) = count.toDouble / wordCount
    }
    }

    bigramCounts.foreach { case (word: Tuple2[Int, Int], count: Int) => {
      if (word._1 == dictionary(Start)) {
        bigramProbability(word) = count.toDouble / lineCount
      }
      else {
        bigramProbability(word) = count.toDouble / unigramCounts(word._1).toDouble
      }
      bigramHistoryProbablity(word._1) = bigramHistoryProbablity.getOrElse(word._1, 0.0) + unigramProbability(word._2)
      bigramHistoryCount(word._1) = bigramHistoryCount.getOrElse(word._1, 0) + 1
    }
    }

    trigramCounts.foreach { case (word: Tuple3[Int, Int, Int], count: Int) => {
      if (word._1 == dictionary(Start) && word._2 == dictionary(Start)) {
        trigramProbability(word) = count.toDouble / lineCount
      }
      else {
        trigramProbability(word) = count.toDouble / bigramCounts(Tuple2(word._1, word._2)).toDouble
      }
      if (trigramHistorySet.contains(Tuple2(word._1, word._2))) {
        trigramHistorySet(Tuple2(word._1, word._2)) += word._3
      }
      else {
        trigramHistorySet(Tuple2(word._1, word._2)) = mutable.Set(word._3)
      }
    }
    }

    Tuple6(unigramProbability.toMap, bigramProbability.toMap, trigramProbability.toMap, bigramHistoryProbablity.toMap, bigramHistoryCount.toMap, trigramHistorySet.toMap)
  }

}


object LinearInterpolation extends App {

  /**
    * This method calculates score for a word by interpolating probability estimates from all n-gram models.
    * p(w|u,v) = λ1p(w|u,v) + λ2p(w|v) + λ3p(w)
    *
    * @param dictionary      - Words in vocab (v)
    * @param heldOutCorpus   - Corpus to be tested
    * @param nGramParameters - It has unigram, bigram, trigram probabilities, bigram history proability, bigram history count and trigram gram history set.
    *                        For eg: unigram probability will have unigram probabilities, bigram will have bigram probabilities, trigram will have trigram probabilities,
    *                        bigram history probability will have sum of probabilities of bigrams that have history h in bigrams, bigram history count has number of words that have history h
    *                        and trigram history set is set of words that have history h in trigram
    * @param hyperParameters - Weights for trigram, bigram and unigram (λ1,λ2,λ3)
    * @return - Entropy for every sentence in testing corpus.
    */

  def linearInterpolation(dictionary: Map[String, Int],
                          heldOutCorpus: List[String],
                          nGramParameters: Tuple6[Map[Int, Double], Map[Tuple2[Int, Int], Double], Map[Tuple3[Int, Int, Int], Double], Map[Int, Double], Map[Int, Int], Map[Tuple2[Int, Int], mutable.Set[Int]]],
                          hyperParameters: Tuple3[Double, Double, Double]): Map[Int, Double] = {

    /**
      * Map for storing entropies for every sentence
      */
    val sentenceEntropies = mutable.Map[Int, Double]()

    heldOutCorpus.zipWithIndex.foreach({ case (line: String, lineNo: Int) => {

      var sentenceEntropy = 0.0

      val words = (dictionary(Common.Start) + " " + dictionary(Common.Start) + " " + line + " " + dictionary(Common.Stop)).split("[\\s]+")
      words.zipWithIndex.foreach { case (word: String, position: Int) => {

        if (position > 1) {

          /*
            Probability estimate of a word using trigram model.
             */
          val trigramScore: Double = (hyperParameters._1 * nGramParameters._3.getOrElse((Tuple3(words(position - 2).toInt, words(position - 1).toInt, word.toInt)), 0.0))
          /*
          Probability estimate of a word using bigram model.
           */
          val bigramScore: Double = (hyperParameters._2 * nGramParameters._2.getOrElse(Tuple2(words(position - 1).toInt, word.toInt), 0.0))
          /*
           Probability estimate of a word using bigram model.
           */
          val unigramScore: Double = (hyperParameters._3 * nGramParameters._1(word.toInt))

          /*
            Score of word is taking log of interpolation of probability estimates from unigram, bigram and trigram
           */
          val totalScore: Double = trigramScore + bigramScore + unigramScore
          /*
          Adding score to sentence entropy
           */
          sentenceEntropy = sentenceEntropy + (Math.log(totalScore) / Math.log(2.0))
        }
      }
      }
      sentenceEntropies(lineNo) = sentenceEntropy
    }
    })
    /*
    Return sentence entropt
     */
    sentenceEntropies.toMap

  }

  /**
    * This method calculates perplexity of testing corpus and is called from [[]]
    *
    * @param dictionary      - List of words in vocab
    * @param heldOutCorpus   - Testing corpus
    * @param nGramParameters - It has unigram, bigram, trigram probabilities, bigram history proability, bigram history count and trigram gram history set.
    *                        For eg: unigram probability will have unigram probabilities, bigram will have bigram probabilities, trigram will have trigram probabilities,
    *                        bigram history probability will have sum of probabilities of bigrams that have history h in bigrams, bigram history count has number of words that have history h
    *                        and trigram history set is set of words that have history h in trigram
    * @param hyperParameters - Weights for trigram, bigram and unigram (λ1,λ2,λ3)
    * @return - Returns perplexity of the testing corpus
    */

  def computePerplexity(dictionary: Map[String, Int],
                        heldOutCorpus: List[String],
                        nGramParameters: Tuple6[Map[Int, Double], Map[Tuple2[Int, Int], Double], Map[Tuple3[Int, Int, Int], Double], Map[Int, Double], Map[Int, Int], Map[Tuple2[Int, Int], mutable.Set[Int]]],
                        hyperParameters: Tuple3[Double, Double, Double]): Double = {
    /**
      * Stores sentence entropy
      */
    val (sentenceEntropies): Map[Int, Double] = linearInterpolation(dictionary, heldOutCorpus, nGramParameters, hyperParameters)

    /*
        Calculate number of words in testing corpus
     */
    var heldOutCorpusWordCount: Double = 0
    heldOutCorpus.foreach((line: String) => {

      line.split("[\\s]+").foreach((word: String) => {
        heldOutCorpusWordCount = heldOutCorpusWordCount + 1.0
      })

    })
    /*
    Calculating and returning perplexity. Perplexity is given by
     */
    Math.pow(2, sentenceEntropies.values.sum / heldOutCorpusWordCount * -1)
  }

  def run() = {
    println("LinearInterpolation")
    println("Enter experiment configuration as: training,testing,y1,y2,y3 for example: brown,brown,0.1,0.5,0.4")
    val input = readLine()
    val inputParams = input.split(",")

    val (trainingCorpus, devTrainingCorpus, testingCorpus, dictionary, nGramCounts, nGramProbablities) = Common.loadCorpus(inputParams(0), inputParams(1))
    val perplexity: Double = computePerplexity(dictionary, testingCorpus, nGramProbablities, Tuple3(inputParams(2).toDouble, inputParams(3).toDouble, inputParams(4).toDouble))
    println(inputParams(0) + "," + inputParams(1) + "," + inputParams(2) + "," + inputParams(3) + "," + inputParams(4) + "," + perplexity + "\n")
  }

  run()

}
import scala.collection.mutable

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


object Backoff extends App {

  /**
    * This method is used to calculate bigram backoff probability.
    *
    * @param dictionary              - List of words in vocab
    * @param bigram                  - Bigram whose bigram backoff probability needs to be calulated
    * @param nGramCounts             - unigram, bigram and trigram counts for ngrams in vocab
    * @param nGramParameters         - It has unigram, bigram, trigram probabilities, bigram history proability, bigram history count and trigram gram history set.
    *                                For eg: unigram probability will have unigram probabilities, bigram will have bigram probabilities, trigram will have trigram probabilities,
    *                                bigram history probability will have sum of probabilities of bigrams that have history h in bigrams, bigram history count has number of words that have history h
    *                                and trigram history set is set of words that have history h in trigram
    * @param backoffDiscount         -  hyperparameter discount count
    * @param trainingCorpusLineCount - Number of lines in trainingcorpus
    * @return - Returns bigram backoff probability of a bigram.
    */

  def computeBigramBackOffProbablity(dictionary: Map[String, Int],
                                     bigram: Tuple2[Int, Int],
                                     nGramCounts: Tuple3[Map[Int, Int], Map[Tuple2[Int, Int], Int], Map[Tuple3[Int, Int, Int], Int]],
                                     nGramParameters: Tuple6[Map[Int, Double], Map[Tuple2[Int, Int], Double], Map[Tuple3[Int, Int, Int], Double], Map[Int, Double], Map[Int, Int], Map[Tuple2[Int, Int], mutable.Set[Int]]],
                                     backoffDiscount: Double,
                                     trainingCorpusLineCount: Int): Double = {
    /*
      Check if it is first word of line
     */
    val historyCount = if (dictionary(Common.Start) == bigram._1)
      trainingCorpusLineCount
    else
      nGramCounts._1(bigram._1)

    /**
      * Part of Set A - Words that have history.
      */
    if (nGramCounts._2.contains(bigram)) {
      return (nGramCounts._2(bigram) - backoffDiscount) / historyCount
    }

    /**
      * Part of Set B - Words that don't have history
      */
    else {

      /*
        Missing mass function is alpha = discount * number of words with history / count of history
       */
      val alpha: Double = (backoffDiscount * nGramParameters._5(bigram._1)) / historyCount
      /*
      Calculating unigram probability of a word
       */
      val numerator: Double = nGramParameters._1(bigram._2)
      val denominator: Double = 1.0 - nGramParameters._4(bigram._1)

      return alpha * (numerator / denominator)
    }

  }

  /**
    * This method calculates trigram backoff probability of word
    *
    * @param dictionary              - List of words in dictionary
    * @param trigram                 - Trigram whose backoff needs to be calculated
    * @param nGramCounts             - Unigram, bigram and trigram counts
    * @param nGramParameters         -  It has unigram, bigram, trigram probabilities, bigram history proability, bigram history count and trigram gram history set.
    *                                For eg: unigram probability will have unigram probabilities, bigram will have bigram probabilities, trigram will have trigram probabilities,
    *                                bigram history probability will have sum of probabilities of bigrams that have history h in bigrams, bigram history count has number of words that have history h
    *                                and trigram history set is set of words that have history h in trigram
    * @param backoffDiscount         - Discount factor
    * @param trainingCorpusLineCount - Number of lines in training corpus
    * @return - Returns trigram backoff probability of a word.
    */

  def computeTrigramBackOffProbablity(dictionary: Map[String, Int],
                                      trigram: Tuple3[Int, Int, Int],
                                      nGramCounts: Tuple3[Map[Int, Int], Map[Tuple2[Int, Int], Int], Map[Tuple3[Int, Int, Int], Int]],
                                      nGramParameters: Tuple6[Map[Int, Double], Map[Tuple2[Int, Int], Double], Map[Tuple3[Int, Int, Int], Double], Map[Int, Double], Map[Int, Int], Map[Tuple2[Int, Int], mutable.Set[Int]]],
                                      backoffDiscount: Double,
                                      trainingCorpusLineCount: Int): Double = {
    /*
    Check if it is first word of a line
     */
    val historyCount = if (dictionary(Common.Start) == trigram._1 && dictionary(Common.Start) == trigram._2)
      trainingCorpusLineCount
    else
      nGramCounts._2.getOrElse(Tuple2(trigram._1, trigram._2), 0)

    /**
      * Part of Set A - Words that have history
      */
    if (nGramCounts._3.contains(trigram)) {
      return (nGramCounts._3(trigram) - backoffDiscount) / historyCount
    }

    /**
      * Part of Set B - Words that don't have history
      */
    else {
      /*
      Set of words in vocab that have history trigram._1 and trigram._2
       */
      val wordsWithHistory: Set[Int] = nGramParameters._6.getOrElse(Tuple2(trigram._1, trigram._2), Set.empty).toSet

      /**
        * Missing mass function is alpha = discount * number of words with history / count of history if history is present in testing corpus else it is 1
        */
      val alpha: Double = if (historyCount == 0) 1.0 else (backoffDiscount * wordsWithHistory.size) / historyCount

      /**
        * Calculating backoff bigram probability
        */
      val numerator: Double = computeBigramBackOffProbablity(dictionary, Tuple2(trigram._2, trigram._3), nGramCounts, nGramParameters, backoffDiscount, trainingCorpusLineCount)
      /**
        * Calculating sum of backoff probability of words that don't have history which is equal to (1 - sum of probability of words that have history)
        */
      val denominator: Double = 1.00 - wordsWithHistory.map(word => computeBigramBackOffProbablity(dictionary, Tuple2(trigram._2, word), nGramCounts, nGramParameters, backoffDiscount, trainingCorpusLineCount)).sum

      return (alpha * (numerator / denominator))
    }
  }

  /**
    *
    * @param dictionary              - List of words in vocab
    * @param heldOutCorpus           - Testing corpus to be tested
    * @param nGramCounts             - unigram, bigram and trigram count for words in vocab
    * @param nGramParameters         - It has unigram, bigram, trigram probabilities, bigram history proability, bigram history count and trigram gram history set.
    *                                For eg: unigram probability will have unigram probabilities, bigram will have bigram probabilities, trigram will have trigram probabilities,
    *                                bigram history probability will have sum of probabilities of bigrams that have history h in bigrams, bigram history count has number of words that have history h
    *                                and trigram history set is set of words that have history h in trigram
    * @param backoffDiscount         - Discount factor
    * @param trainingCorpusLineCount - Number of lines in training corpus
    * @return
    */
  def backoff(dictionary: Map[String, Int],
              heldOutCorpus: List[String],
              nGramCounts: Tuple3[Map[Int, Int], Map[Tuple2[Int, Int], Int], Map[Tuple3[Int, Int, Int], Int]],
              nGramParameters: Tuple6[Map[Int, Double], Map[Tuple2[Int, Int], Double], Map[Tuple3[Int, Int, Int], Double], Map[Int, Double], Map[Int, Int], Map[Tuple2[Int, Int], mutable.Set[Int]]],
              backoffDiscount: Double,
              trainingCorpusLineCount: Int) = {
    /**
      * Stores entropies for a sentence in corpus
      */
    val sentenceEntropies = mutable.Map[Int, Double]()

    heldOutCorpus.zipWithIndex.foreach({ case (line: String, lineNo: Int) => {
      var sentenceEntropy = 0.0

      val words = (dictionary(Common.Start) + " " + dictionary(Common.Start) + " " + line + " " + dictionary(Common.Stop)).split("[\\s]+")
      words.zipWithIndex.foreach { case (word: String, position: Int) => {

        if (position > 1) {
          /**
            * Calculates backoff probability fo a word
            */
          val score: Double = computeTrigramBackOffProbablity(dictionary, Tuple3(words(position - 2).toInt, words(position - 1).toInt, word.toInt), nGramCounts, nGramParameters, backoffDiscount, trainingCorpusLineCount)

          /**
            * Calculates entropy of a sentence
            */
          sentenceEntropy = sentenceEntropy + (Math.log(score) / Math.log(2.0))
        }
      }
      }

      //Returns entropy of every sentence
      sentenceEntropies(lineNo) = sentenceEntropy
    }
    })

    sentenceEntropies.toMap

  }

  /**
    * This method computes perplexity of the corpus.
    *
    * @param dictionary              - List of words in dictionary
    * @param heldOutCorpus           - Testing corpus
    * @param nGramCounts             - Unigram, bigram, trigram counts
    * @param nGramParameters         - It has unigram, bigram, trigram probabilities, bigram history proability, bigram history count and trigram gram history set.
    *                                For eg: unigram probability will have unigram probabilities, bigram will have bigram probabilities, trigram will have trigram probabilities,
    *                                bigram history probability will have sum of probabilities of bigrams that have history h in bigrams, bigram history count has number of words that have history h
    *                                and trigram history set is set of words that have history h in trigram
    * @param backoffDiscount         - Discount factor
    * @param trainingCorpusLineCount - Number of lines in training corpus
    * @return Returns perplexity of the corpus
    */
  def computePerplexity(dictionary: Map[String, Int],
                        heldOutCorpus: List[String],
                        nGramCounts: Tuple3[Map[Int, Int], Map[Tuple2[Int, Int], Int], Map[Tuple3[Int, Int, Int], Int]],
                        nGramParameters: Tuple6[Map[Int, Double], Map[Tuple2[Int, Int], Double], Map[Tuple3[Int, Int, Int], Double], Map[Int, Double], Map[Int, Int], Map[Tuple2[Int, Int], mutable.Set[Int]]],
                        backoffDiscount: Double,
                        trainingCorpusLineCount: Int): Double = {
    val sentencePerplexities: Map[Int, Double] = backoff(dictionary, heldOutCorpus, nGramCounts, nGramParameters, backoffDiscount, trainingCorpusLineCount)

    var heldOutCorpusWordCount: Double = 0
    heldOutCorpus.foreach((line: String) => {
      line.split("[\\s]+").foreach((word: String) => {
        heldOutCorpusWordCount = heldOutCorpusWordCount + 1.0
      })
    })

    Math.pow(2, (sentencePerplexities.values.sum / heldOutCorpusWordCount) * -1)
  }

  def run() = {
    println("Backoff")
    println("Enter experiment conifguration as: training,testing,backOffDiscount, for example: brown,brown,0.5")
    val input = readLine()
    val inputParams = input.split(",")

    val (trainingCorpus, devTrainingCorpus, testingCorpus, dictionary, nGramCounts, nGramProbablities) = Common.loadCorpus(inputParams(0), inputParams(1))

    val perplexity: Double = computePerplexity(dictionary, testingCorpus, nGramCounts, nGramProbablities, inputParams(2).toDouble, trainingCorpus.length)
    println(inputParams(0) + "," + inputParams(1) + "," + inputParams(2) + "," + perplexity + "\n")
  }

  run()

}

import getopt
import sys
import pdb
import time
import re
import os
import numpy as np
import scipy.linalg
# import for using gammaln, psi
from scipy.special import *
from scipy.spatial.distance import cdist
import mpmath
from utils import *

# T: means of the vmf's, mu in the paper
class vmfMix:
    def __init__(self, **kwargs):
        self.K = kwargs.get( 'K', 30 )
        # hyperparameter of the Dirichlet
        self.alpha = kwargs.get( 'alpha', 0.5 )
        self.MAX_EM_ITERS = kwargs.get( 'MAX_EM_ITERS', 100 )
        self.TDiff_tolerance = kwargs.get( 'TDiff_tolerance', 2e-3 )
        # maximum difference between kappa's of different components
        self.max_kappa_diff = kwargs.get( 'max_kappa_diff', 400 )
        self.seed = kwargs.get( 'seed', 0 )
        self.verbose = kwargs.get( 'verbose', 1 )

        # print T every so many iters, for debugging
        self.printTopics_iterNum = kwargs.get( 'printTopics_iterNum', 20 )
        # number of top tags to output into logfile
        self.topTags = kwargs.get( 'topTags', 12 )
        # output only the first 'topDim' dimensions of T
        self.topDim = kwargs.get( 'topDim', 10 )
        # print top topics with top "words" in these topics
        self.topTopicMassFracPrintThres = kwargs.get( 'topTopicMassFracPrintThres', 1 )

        if 'fileLogger' not in kwargs:
            self.logfilename = kwargs.get( 'logfilename', "vmfMix" )
            self.fileLogger = initFileLogger( self.logfilename, False )
        else:
            self.fileLogger = kwargs['fileLogger']
        self.ni_k, self.n__k, self.logcd, self.rbar  = None, None, None, None
        self.Pi, self.Fi, self.T, self.kappa = None, None, None, None
        self.corpusName, self.X, self.L, self.Tags, self.r = None, None, None, None, None
        self.D, self.M, self.it, self.totalL = 0, 0, 0, 0
        # topic assignment by kmeans for comparison
        self.kmeans_xtoc = self.kmeans_distances = None
        self.evalKmeans = kwargs.get( 'evalKmeans', False )

    def calc_n(self):
        # n_{i.k}: document level effective point counts. M * K
        self.ni_k = np.zeros((self.M, self.K))
        for i in xrange(self.M):
            self.ni_k[i] = np.sum( self.Pi[i] * self.Freqs[i][:,None], axis=0 )
        # n_{..k}: corpus level effective point counts. K-dim vector
        self.n__k = np.sum( self.ni_k, axis = 0 )

    # r: sum_ij pi_ij. * x_ij. shape: K * D
    def calc_r(self):
        self.r = np.zeros( (self.K, self.D) )
        for i in xrange(self.M):
            # Pi[i]: Li * K, X[i]: Li * D. pi_i_x_i: K * D
            pi_i_x_i = np.dot( (self.Pi[i] * self.Freqs[i][:,None]).T, self.X[i] )
            self.r += pi_i_x_i

        # rbar: K-dim vector
        self.rbar = scipy.linalg.norm(self.r, axis=1)
        self.rbar /= self.n__k

    def calc_logcd(self):
        d_2_1 = self.D/2 - 1
        self.logcd = d_2_1 * np.log(self.kappa) - self.D/2 * np.log( np.pi * 2 )
        for k in xrange(self.K):
            self.logcd[k] -= float( mpmath.log( mpmath.besseli(d_2_1, self.kappa[k]) ) )
        # logcd2 is equivalent to logcd (after normalization) when doing inference, 
        # but avoids being too big exponents
        self.logcd2 = self.logcd - np.min(self.logcd)
        
    # M-step
    def updateT_kappa(self):
        oldT = self.T
        # r, T: K * D
        self.T = normalizeF(self.r)
        # rbar, kappa: K-dim vector. rbar is a copy of self.rbar
        rbar = self.rbar + 0
        # Sometimes rbar=1 (a very rare word in embedding training corpus, but not rare in topic training corpus
        # which results in kappa=infinity :(
        # rbar <= 0.95  ->  kappa < 5200 when D=500. 
        rbar[rbar > 0.95] = 0.95
        self.kappa = ( self.D * rbar - np.power( rbar, 3 ) ) / ( 1 - np.power( rbar, 2 ) )
        # maximal allowed kappa
        max_kappa = np.min(self.kappa) + self.max_kappa_diff
        # cap excessively big kappa to max_kappa
        self.kappa[ self.kappa > max_kappa ] = max_kappa
        self.calc_logcd()
        TDiff = self.T - oldT
        # max_tdiff: K-dim vector
        max_tdiff = np.max( np.linalg.norm( TDiff, axis=1 ) )
        TDiffNorm = np.linalg.norm(TDiff)
        return max_tdiff, TDiffNorm

    # Pi: list of M arrays. Pi[i]: L[i] * K
    def updatePi(self):
        psiFi = psi(self.Fi)
        # different docs may have different lengths, so Pi can't be a numpy array
        self.Pi = []
        for i in xrange(self.M):
            # X[i]: Li * D. T: K * D.
            # vec * Mat: vec * each row of Mat
            # i.e. kappa * each row of XT. kappa_XT: Li * K
            kappa_XT = self.kappa * np.dot( self.X[i], self.T.T )
            # pi_i: Li * K
            try:
                pi_i = np.exp( psiFi[i] + self.logcd2 + kappa_XT )
            except FloatingPointError, e:
                pdb.set_trace()
                
            if np.sum(pi_i) < 1e-6:
                pdb.set_trace()
            pi_i = normalize(pi_i)
            self.Pi.append(pi_i)

    # Fi: M * K
    def updateFi(self):
        self.Fi = self.ni_k + self.alpha

    def calcVarLB(self):
        totalVarLB = 0

        for i in xrange(self.M):
            # K-dim vector
            fi_i = self.Fi[i]
            # pi_i & freq_pi_i: Li * K. X[i]: Li * D. T: K * D
            pi_i = self.Pi[i]
            freq_pi_i = pi_i * self.Freqs[i][:,None]

            fi_i0 = np.sum(fi_i)
            entropy = np.sum( gammaln(fi_i) ) - gammaln(fi_i0)
            entropy += (fi_i0 - self.K) * psi(fi_i0) - np.sum( (fi_i - 1) * psi(fi_i) )
            entropy -= np.sum( freq_pi_i * np.log(pi_i) )
            # ni_k: M * K. ni_k_fi_i: K-dim vector
            ni_k_fi_i = ( self.ni_k[i] + self.alpha - 1 ) * ( psi(fi_i) - psi(fi_i0) )
            loglike = entropy + np.sum(ni_k_fi_i)

            totalVarLB += loglike

        # sum of log of normalization constants
        vmfNorm = np.sum( self.logcd * self.n__k )
        # r is the weighted sum of X
        totalVarLB += vmfNorm + np.trace( self.kappa * np.dot( self.T, self.r.T ) )
        return totalVarLB

    # the returned outputter always output to the log file
    # screenVerboseThres controls when the generated outputter will output to screen
    # when self.verbose >= screenVerboseThres, screen output is enabled
    # in the batch mode for multiple files, typically self.verbose == 0
    # then by default no screen output anyway
    # in the single file mode, typically self.verbose == 1
    # then in out0, screenVerboseThres = 0
    # => always with screen output
    # in out1, screenVerboseThres = 1
    # => if verbose =0 < screenVerboseThres, no screen output
    def genOutputter(self, screenVerboseThres=1):
        def screen_log_output(s):
            self.fileLogger.debug(s)
            if self.verbose >= screenVerboseThres:
                print s
        return screen_log_output

    def setX(self, corpusName, X, Freqs=None, Tags=None):
        self.corpusName = corpusName
        # VMF mixture is done on the unit hyper-sphere. So do normalization first
        self.X = []
        for Xi in X:
            self.X.append( normalizeF(Xi) )
        # number of documents
        self.M = len(self.X)
        # X[i] and X[j] may have different lengths. Pick any one, e.g. X[0]
        # dimensionality of embeddings
        self.D = self.X[0].shape[1]
        # Freqs, Tags have the same shape as the first two dimensions of X
        # Freqs[i][j]: freq of tag Tags[i][j] in document i.
        # if Freqs is None: all Freqs are set to one, 
        # i.e. all tags appear only once in the current document
        # this is the case when X is image neural encodings. 
        # Update equations will fall back to the exact equations in the vmf-mixture document.
        # if X is word embeddings in documents, then Freqs should be word frequencies
        self.Tags = Tags
        if Freqs is None:
            self.Freqs = []
            for i in xrange(self.M):
                self.Freqs.append( np.ones(X[i].shape[0]) )
        else:
            self.Freqs = Freqs
        self.L = np.zeros(self.M)
        # L[i]: length (number of vectors) of the i-th doc
        for i in xrange(self.M):
            self.L[i] = np.sum(self.Freqs[i])
        self.totalL = np.sum(self.L)

    # topTopicMassFracPrintThres: when a topic's fraction n__k/totalL > topTopicMassFracPrintThres/K, print it
    def printTopTagsInTopics(self, outputToScreen=False):
        # the topic prop of each tag, stored in separate dicts by topics
        topic_tag2prop = [ {} for i in xrange(self.K) ]
        # tag occurrences
        tag2occur = {}
        # map tag to the vector
        tag2x = {}
        for i in xrange(self.M):
            for j in xrange( self.X[i].shape[0] ):
                if self.Tags is None:
                    tag = '%d-%d' %(i,j)
                else:
                    tag = self.Tags[i][j]

                if self.evalKmeans:
                    k = self.kmeans_xtoc[j]
                    topic_tag2prop[k][tag] += 1
                else:
                    for k in xrange(self.K):
                        # a tag may appear multiple times, e.g. words in documents
                        # Sum up the proportions of all occurrences
                        if tag in topic_tag2prop[k]:
                            topic_tag2prop[k][tag] += self.Pi[i][j,k] * self.Freqs[i][j]
                        else:
                            topic_tag2prop[k][tag] = self.Pi[i][j,k] * self.Freqs[i][j]

                if tag in tag2occur:
                    tag2occur[tag] += self.Freqs[i][j]
                else:
                    tag2occur[tag] = self.Freqs[i][j]

                tag2x[tag] = self.X[i][j]
                
        if self.evalKmeans:
            n__k = np.bincount(self.kmeans_xtoc)
        else:
            n__k = self.n__k

        # tids is sorted topic IDs from most frequent to least frequent
        tids = sorted( range(self.K), key=lambda k: n__k[k], reverse=True )
        for i,k in enumerate(tids):
            # below the average proportion * topTopicMassFracPrintThres
            if n__k[k] < self.topTopicMassFracPrintThres * self.totalL / self.K:
                break

        # cut_i is the cut point of tids: tids[:cut_i] will be printed
        # if i==0, i.e. all topics have proportions < threshold
        # this may happen when topicThres is too big.
        # in this case, print the biggest topic
        cut_i = max(1, i)

        # the topic prop of each word, indexed by the row ID
        # take account of the word freq, but dampen it with sqrt
        # so that more similar, less frequent words have chance to be selected
        # doing average does not consider freq, not good either
        topic_tag2dampedProp = [ {} for i in xrange(self.K) ]
        for k in xrange(self.K):
            for tag, prop in topic_tag2prop[k].iteritems():
                topic_tag2dampedProp[k][tag] = prop / np.sqrt(tag2occur[tag])

        if outputToScreen:
            out = self.genOutputter(1)
        else:
            out = self.genOutputter(2)

        out("")
        out( "n__k:\n%s\n" %n__k )
        if not self.evalKmeans:
            # ni_k of the first 5 documents as samples
            out( "ni_k[:5]:\n%s\n" %self.ni_k[:5] )
        
        if not self.evalKmeans:
            # r shape: K * D
            out("r[:,:10]:\n%s\n" %self.r[:,:10])

        # selected tids to output
        selTids = tids[:cut_i]
        selTids = np.array(selTids)

        for k in selTids:
            tag2dampedProp = topic_tag2dampedProp[k]
            out( "Topic %d: %.1f%%" %( k, 100 * n__k[k] / self.totalL ) )

            tags_sorted = sorted( tag2dampedProp.keys(), key=lambda tag: tag2dampedProp[tag], reverse=True )

            out("Most relevant tags:")

            line = ""
            for tag in tags_sorted[:self.topTags]:
                topicDampedProp = tag2dampedProp[tag]
                topicProp = topic_tag2prop[k][tag]
                dotprod = np.dot( tag2x[tag], self.T[k] )

                line += "%s (%d): %.2f/%.2f/%.2f " %( tag, tag2occur[tag],
                                    topicDampedProp, topicProp, dotprod )

            out(line)
            out("")

    def inferTopicProps( self, T, kappa, MAX_ITERS=5 ):
        self.T = T
        self.kappa = kappa
        # uniform prior
        self.Fi = np.ones( (self.M, self.K) )
        loglike = 0
        self.calc_logcd()
        
        for i in xrange(MAX_ITERS):
            iterStartTime = time.time()
            Pi2 = self.Pi
            self.updatePi()
            self.calc_n()
            self.updateFi()

            if i > 0:
                Pi_diff = np.zeros(self.M)
                for i in xrange(self.M):
                    Pi_diff[i] = np.linalg.norm( self.Pi[i] - Pi2[i] )
                max_Pi_diff = np.max(Pi_diff)
                total_Pi_diff = np.sum(Pi_diff)
            else:
                max_Pi_diff = 0
                total_Pi_diff = 0

            iterDur = time.time() - iterStartTime
            print "Iter %d loglike %.2f, Pi diff total %.3f, max %.3f. %.1fs" %( i, 
                                 loglike, total_Pi_diff, max_Pi_diff, iterDur )

        return self.ni_k, self.Pi

    def inference(self):
        startTime = time.time()
        startTimeStr = timeToStr(startTime)

        # out0 prints both to screen and to log file, regardless of the verbose level
        out0 = self.genOutputter(0)
        out1 = self.genOutputter(1)
        out0( "%d topics." %(self.K) )
        out0( "%s inference starts at %s" %( self.corpusName, startTimeStr ) )

        self.T = np.zeros( ( self.K, self.D ) )
        self.kappa = np.zeros(self.K)

        if self.seed != 0:
            np.random.seed(self.seed)
            out0( "Seed: %d" %self.seed )

        for k in xrange(0, self.K):
            self.T[k] = np.random.multivariate_normal( np.zeros(self.D), np.eye(self.D) )
            self.T[k] = normalizeF(self.T[k])
            self.kappa[k] = np.random.lognormal(1)

        # update log c_d(kappa)
        self.calc_logcd()
        # Fi is initialized as 1 (uniform, not considering ni_k)
        self.Fi = np.ones( (self.M, self.K) )

        lastIterEndTime = time.time()
        self.updatePi()
        self.calc_n()
        self.updateFi()
        self.calc_r()
        varLB = self.calcVarLB()

        self.it = 0
        iterDur = time.time() - lastIterEndTime
        lastIterEndTime = time.time()

        print "Iter %d: loglike %.2f, %.1fs" %( self.it, varLB, iterDur )

        endTime = time.time()
        endTimeStr = timeToStr(endTime)
        inferDur = int(endTime - startTime)
        Tkappa_loglikes = []
        
        # an arbitrary number to satisfy pylint
        TDiffNorm = 100000
        while self.it == 0 or ( self.it < self.MAX_EM_ITERS and TDiffNorm > self.TDiff_tolerance ):
            self.it += 1
            self.fileLogger.debug( "EM Iter %d:", self.it )
            max_tdiff, TDiffNorm = self.updateT_kappa()
            self.updatePi()
            self.calc_n()
            self.updateFi()
            self.calc_r()

            varLB2 = varLB
            varLB = self.calcVarLB()

            iterDur = time.time() - lastIterEndTime
            lastIterEndTime = time.time()

            iterStatusMsg = "Iter %d: loglike %.2f, TDiffNorm %.4f, max_tdiff %.3f, %.1fs" %( self.it,
                                           varLB, TDiffNorm, max_tdiff, iterDur )

            if self.it % self.printTopics_iterNum == 0:
                out0(iterStatusMsg)

                if self.verbose >= 2:
                    self.fileLogger.debug( "T[:,%d]:", self.topDim )
                    self.fileLogger.debug( self.T[ :, :self.topDim ] )

                    self.fileLogger.debug("kappa:")
                    self.fileLogger.debug(self.kappa)
                self.printTopTagsInTopics()
            else:
                # not using out0 because the "\r" in the console output shouldn't be in the log file
                print "%s  \r" %iterStatusMsg,
                self.fileLogger.debug(iterStatusMsg)
                self.fileLogger.debug( "n__k:\n%s\n", self.n__k )
                
            Tkappa_loglikes.append( [ self.it, self.T, self.kappa, varLB ] )
            
        if self.verbose >= 1:
            # if == 0, topics has just been printed in the while loop
            if self.it % self.printTopics_iterNum != 0:
                self.printTopTagsInTopics()

        endTime = time.time()
        endTimeStr = timeToStr(endTime)
        inferDur = int(endTime - startTime)

        print
        out0( "%s inference ends at %s. %d iters, %d seconds." %( self.corpusName, endTimeStr, self.it, inferDur ) )

        # sort according to varLB
        Tkappa_loglikes_sorted = sorted( Tkappa_loglikes, key=lambda Tkappa_loglikes: Tkappa_loglikes[3], reverse=True )
        # best T could be the last T. 
        # In that case, the two elements in best_last_Tkappa are the same
        best_last_Tkappa = [ Tkappa_loglikes_sorted[0], Tkappa_loglikes[-1] ]

        return best_last_Tkappa, self.ni_k

    # kmeans is only for a single document
    def kmeans( self, maxiter=10 ):
        """ centers, Xtocentre, distances = topicvec.kmeans( ... )
        in:
            X: M x D
            centers K x D: initial centers, e.g. random.sample( X, K )
            iterate until the change of the average distance to centers
                is within TDiff_tolerance of the previous average distance
            maxiter
            metric: cosine
            self.verbose: 0 silent, 2 prints running distances
        out:
            centers: K x D
            xtoc: each x -> its nearest center, ints M -> K
            distances: M
        """

        X = self.X[0]
        centers = randomsample( X, self.K )

        if self.verbose:
            print "kmeans: X %s  centers %s  tolerance=%.2g  maxiter=%d" %(
                X.shape, centers.shape, self.TDiff_tolerance, maxiter )

        L = X.shape[0]
        allx = np.arange(L)
        prevdist = 0

        for it in range( 1, maxiter+1 ):
            Dists_xc = cdist( X, centers, metric='cosine' )  # |X| x |centers|
            xtoc = Dists_xc.argmin(axis=1)  # X -> nearest center
            distances = Dists_xc[ allx, xtoc ]
            #avgdist = distances.mean()  # median ?
            avgdist = distances.mean()

            if self.verbose >= 2:
                print "kmeans: avg |X - nearest center| = %.4g" % avgdist

            if (1 - self.TDiff_tolerance) * prevdist <= avgdist <= prevdist \
            or it == maxiter:
                break

            prevdist = avgdist

            for k in range(self.K):  # (1 pass in C)
                # np.where(..)[0]: array of indices
                c = np.where( xtoc == k )[0]
                if len(c) > 0:
                    centers[k] = X[c].mean( axis=0 )
                    centers[k] = normalizeF(centers[k])

        if self.verbose:
            print "kmeans: %d iterations.  Cluster sizes:" % it, np.bincount(xtoc)

        if self.verbose >= 2:
            r50 = np.zeros(self.K)
            r90 = np.zeros(self.K)
            for k in range(self.K):
                dist = distances[ xtoc == k ]
                if len(dist) > 0:
                    r50[k], r90[k] = np.percentile( dist, (50, 90) )
            print "kmeans: cluster 50% radius", r50.astype(int)
            print "kmeans: cluster 90% radius", r90.astype(int)

        self.T = centers
        self.kmeans_xtoc = xtoc
        self.kmeans_distances = distances

import fs from 'fs';
import path from 'path';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeKatex from 'rehype-katex';
import rehypeStringify from 'rehype-stringify';
import rehypeHighlight from 'rehype-highlight';
import rehypeSlug from 'rehype-slug';

interface TreeNode {
  type: string;
  tagName?: string;
  value?: string;
  depth?: number;
  url?: string;
  properties?: Record<string, unknown>;
  children?: TreeNode[];
}

export interface Heading { id: string; text: string; level: number }
export interface RenderedMarkdown { html: string; headings: Heading[]; readingMinutes: number }

function walk(node: TreeNode, visit: (node: TreeNode) => void) {
  visit(node);
  node.children?.forEach(child => walk(child, visit));
}

function nodeText(node: TreeNode): string {
  return node.value ?? node.children?.map(nodeText).join('') ?? '';
}

// Existing Markdown remains unchanged; the article template owns the page H1.
function normalizeHeadings(options: { minimum: number }) {
  return (tree: unknown) => {
    const root = tree as TreeNode;
    let minimum = 6;
    walk(root, node => { if (node.type === 'heading' && node.depth) minimum = Math.min(minimum, node.depth); });
    const shift = options.minimum - minimum;
    let previous = options.minimum - 1;
    walk(root, node => {
      if (node.type === 'heading' && node.depth) {
        node.depth = Math.max(options.minimum, Math.min(6, previous + 1, node.depth + shift));
        previous = node.depth;
      }
      if (node.type === 'image' && node.url?.startsWith('/')) {
        const filename = path.join(process.cwd(), 'public', node.url);
        if (!fs.existsSync(filename)) throw new Error(`Missing Markdown image: ${node.url}`);
      }
    });
  };
}

const imageDimensions: Record<string, [number, number]> = {
  '/images/pca-variance.jpg': [1770, 848],
  '/images/principle-components.jpg': [756, 628],
  '/images/notes/hands-on-ml/end-to-end-project/log-transform.png': [815, 190],
  '/images/notes/hands-on-ml/end-to-end-project/gaussian-rbf.png': [430, 215],
  '/images/notes/hands-on-ml/end-to-end-project/cross-validation.png': [720, 278],
  '/images/notes/hands-on-ml/classification/confusion-matrix.png': [1201, 643],
  '/images/notes/hands-on-ml/classification/decision-thresholds.png': [1498, 498],
  '/images/notes/hands-on-ml/classification/precision-recall-threshold.png': [1921, 921],
  '/images/notes/hands-on-ml/classification/precision-recall-curve.png': [1448, 1003],
  '/images/notes/hands-on-ml/classification/roc-curve.png': [1486, 1158],
  '/images/notes/hands-on-ml/classification/roc-auc-ranking.png': [2416, 566],
  '/images/notes/hands-on-ml/training-linear-models/gradient-descent.png': [1364, 568],
  '/images/notes/hands-on-ml/training-linear-models/feature-scaling.png': [1494, 638],
  '/images/notes/hands-on-ml/training-linear-models/stochastic-gradient-descent.png': [1058, 676],
  '/images/notes/hands-on-ml/training-linear-models/gradient-descent-comparison.png': [1572, 644],
  '/images/notes/hands-on-ml/training-linear-models/learning-curve-underfitting.png': [1354, 768],
  '/images/notes/hands-on-ml/training-linear-models/learning-curve-overfitting.png': [1396, 749],
  '/images/notes/hands-on-ml/training-linear-models/bias-variance-targets.png': [1646, 751],
  '/images/notes/hands-on-ml/training-linear-models/model-flexibility.png': [1146, 741],
  '/images/notes/hands-on-ml/training-linear-models/ridge-regression.png': [1644, 741],
  '/images/notes/hands-on-ml/training-linear-models/lasso-ridge-contours.png': [1268, 1027],
  '/images/notes/hands-on-ml/training-linear-models/regularization-geometry.png': [1760, 759],
  '/images/notes/hands-on-ml/training-linear-models/sigmoid.png': [1884, 533],
  '/images/notes/hands-on-ml/training-linear-models/negative-log-loss.png': [690, 405],
  '/images/notes/hands-on-ml/training-linear-models/logistic-boundary-1d.png': [2026, 667],
  '/images/notes/hands-on-ml/training-linear-models/logistic-boundary-2d.png': [2434, 933],
  '/images/notes/hands-on-ml/training-linear-models/entropy-video.png': [712, 401],
  '/images/notes/hands-on-ml/training-linear-models/softmax-regions.png': [1732, 671],
  '/images/notes/hands-on-ml/decision-trees/iris-tree.png': [913, 733],
  '/images/notes/hands-on-ml/decision-trees/prediction-sketch.png': [940, 460],
  '/images/notes/hands-on-ml/decision-trees/iris-decision-boundaries.png': [1436, 688],
  '/images/notes/hands-on-ml/decision-trees/leaf-probabilities.png': [1111, 891],
  '/images/notes/hands-on-ml/decision-trees/classification-regularization.png': [1850, 718],
  '/images/notes/hands-on-ml/decision-trees/regression-tree.png': [1830, 705],
  '/images/notes/hands-on-ml/decision-trees/regression-depth.png': [2201, 853],
  '/images/notes/hands-on-ml/decision-trees/regression-regularization.png': [2233, 823],
  '/images/notes/hands-on-ml/decision-trees/axis-orientation.png': [2378, 926],
  '/images/notes/hands-on-ml/ensemble-learning/hard-voting.png': [1152, 588],
  '/images/notes/hands-on-ml/ensemble-learning/law-of-large-numbers.png': [1162, 476],
  '/images/notes/hands-on-ml/ensemble-learning/bagging-process.png': [1192, 668],
  '/images/notes/hands-on-ml/ensemble-learning/bagging-code.png': [1418, 316],
  '/images/notes/hands-on-ml/ensemble-learning/bagging-boundaries.png': [2100, 586],
  '/images/notes/hands-on-ml/ensemble-learning/adaboost-reweighting.png': [1620, 992],
  '/images/notes/hands-on-ml/ensemble-learning/adaboost-voting-weight.png': [826, 900],
  '/images/notes/hands-on-ml/ensemble-learning/adaboost-error-emphasis.png': [960, 649],
  '/images/notes/hands-on-ml/ensemble-learning/gradient-boosting-stages.png': [1790, 1943],
  '/images/notes/hands-on-ml/ensemble-learning/stacking-prediction.png': [1240, 717],
  '/images/notes/hands-on-ml/ensemble-learning/stacking-training.png': [1246, 821],
  '/images/notes/hands-on-ml/ensemble-learning/stacking-layers.png': [960, 849],
  '/images/notes/hands-on-ml/ensemble-learning/ensemble-reference-1.png': [1570, 747],
  '/images/notes/hands-on-ml/ensemble-learning/ensemble-reference-2.png': [1570, 583],
  '/images/notes/hands-on-ml/probability-calibration/calibration-curves.png': [1388, 1360],
  '/images/notes/hands-on-ml/dimensionality-reduction/projection.png': [2800, 726],
  '/images/notes/hands-on-ml/dimensionality-reduction/manifold-examples.png': [1450, 610],
  '/images/notes/hands-on-ml/dimensionality-reduction/sphere.png': [658, 508],
  '/images/notes/hands-on-ml/dimensionality-reduction/swiss-roll.png': [1936, 758],
  '/images/notes/hands-on-ml/dimensionality-reduction/decision-boundaries.png': [1436, 1094],
  '/images/notes/hands-on-ml/dimensionality-reduction/pca-variance.png': [1278, 586],
  '/images/notes/hands-on-ml/dimensionality-reduction/sparse-projection-matrix.png': [924, 214],
  '/images/notes/hands-on-ml/dimensionality-reduction/lle-swiss-roll.png': [816, 673],
  '/images/notes/hands-on-ml/dimensionality-reduction/isomap-geodesic.png': [960, 448],
  '/images/notes/hands-on-ml/dimensionality-reduction/isomap-comparison.png': [960, 442],
  '/images/notes/hands-on-ml/dimensionality-reduction/isomap-steps.png': [1124, 454],
  '/images/notes/hands-on-ml/dimensionality-reduction/pca-lda-axes.png': [960, 488],
  '/images/notes/hands-on-ml/dimensionality-reduction/pca-lda-projections.png': [1070, 486],
  '/images/notes/hands-on-ml/dimensionality-reduction/umap-process.png': [1816, 572],
  '/images/notes/hands-on-ml/dimensionality-reduction/methods-summary.png': [960, 480],
  '/images/notes/hands-on-ml/dimensionality-reduction/embedding-comparison.png': [960, 384],
  '/images/notes/hands-on-ml/unsupervised-learning/clustering-data.png': [978, 468],
  '/images/notes/hands-on-ml/unsupervised-learning/kmeans-voronoi.png': [960, 464],
  '/images/notes/hands-on-ml/unsupervised-learning/kmeans-iterations.png': [1576, 662],
  '/images/notes/hands-on-ml/unsupervised-learning/kmeans-plus-plus.png': [960, 618],
  '/images/notes/hands-on-ml/unsupervised-learning/cluster-count.png': [1174, 492],
  '/images/notes/hands-on-ml/unsupervised-learning/inertia-elbow.png': [960, 350],
  '/images/notes/hands-on-ml/unsupervised-learning/silhouette-scores.png': [1360, 380],
  '/images/notes/hands-on-ml/unsupervised-learning/silhouette-diagrams.png': [1238, 1004],
  '/images/notes/hands-on-ml/unsupervised-learning/color-segmentation.png': [960, 636],
  '/images/notes/hands-on-ml/unsupervised-learning/semantic-segmentation.png': [1252, 622],
  '/images/notes/hands-on-ml/unsupervised-learning/instance-segmentation.png': [960, 640],
  '/images/notes/hands-on-ml/unsupervised-learning/dbscan-comparison.png': [960, 358],
  '/images/notes/hands-on-ml/unsupervised-learning/dbscan-neighborhoods.png': [960, 480],
  '/images/notes/hands-on-ml/unsupervised-learning/hdbscan-core-distances.png': [1222, 820],
  '/images/notes/hands-on-ml/unsupervised-learning/hdbscan-mutual-reachability.png': [1272, 822],
  '/images/notes/hands-on-ml/unsupervised-learning/hdbscan-mst.png': [1012, 736],
  '/images/notes/hands-on-ml/unsupervised-learning/hdbscan-hierarchy.png': [1174, 824],
  '/images/notes/hands-on-ml/unsupervised-learning/hdbscan-condensed-tree.png': [1188, 836],
  '/images/notes/hands-on-ml/unsupervised-learning/hdbscan-selected-clusters.png': [1146, 810],
  '/images/notes/hands-on-ml/unsupervised-learning/agglomerative-process.png': [960, 488],
  '/images/notes/hands-on-ml/unsupervised-learning/agglomerative-dendrogram.png': [648, 462],
  '/images/notes/hands-on-ml/unsupervised-learning/mean-shift-preview.png': [960, 677],
  '/images/notes/hands-on-ml/unsupervised-learning/affinity-propagation-preview.png': [534, 401],
  '/images/notes/hands-on-ml/unsupervised-learning/affinity-messages.png': [722, 363],
  '/images/notes/hands-on-ml/unsupervised-learning/gaussian-mixture.png': [960, 505],
  '/images/notes/hands-on-ml/unsupervised-learning/covariance-types.png': [960, 961],
  '/images/notes/hands-on-ml/unsupervised-learning/gmm-information-criteria.png': [960, 305],
  '/images/notes/hands-on-ml/unsupervised-learning/gmm-nonlinear-limitations.png': [960, 339],
  '/images/notes/hands-on-ml/unsupervised-learning/isolation-splits.png': [960, 431],
  '/images/notes/hands-on-ml/unsupervised-learning/isolation-tree.png': [648, 357],
  '/images/notes/hands-on-ml/unsupervised-learning/lof-reachability.png': [530, 721],
  '/images/notes/hands-on-ml/mahalanobis-distance/correlated-features.png': [1126, 448],
  '/images/notes/hands-on-ml/mahalanobis-distance/distance-contours.png': [960, 428],
  '/images/notes/hands-on-ml/mahalanobis-distance/chi-square-distributions.png': [990, 605],
  '/images/notes/hands-on-ml/likelihood-function/probability-likelihood.png': [960, 532],
  '/images/notes/hands-on-ml/likelihood-function/gaussian-laplace-priors.png': [960, 756],
  '/images/notes/hands-on-ml/neural-networks/threshold-logic-unit.png': [960, 422],
  '/images/notes/hands-on-ml/neural-networks/perceptron-layer.png': [960, 468],
  '/images/notes/hands-on-ml/neural-networks/multilayer-perceptron.png': [960, 584],
  '/images/notes/hands-on-ml/neural-networks/activation-functions.png': [960, 444],
  '/images/notes/hands-on-ml/training-deep-networks/sigmoid-saturation.png': [960, 598],
  '/images/notes/hands-on-ml/training-deep-networks/leaky-relu.png': [960, 468],
  '/images/notes/hands-on-ml/training-deep-networks/elu-selu.png': [960, 482],
  '/images/notes/hands-on-ml/training-deep-networks/momentum-intuition.png': [714, 381],
  '/images/notes/hands-on-ml/training-deep-networks/nesterov-momentum.png': [960, 405],
  '/images/notes/hands-on-ml/training-deep-networks/adagrad.png': [960, 479],
  '/images/notes/hands-on-ml/training-deep-networks/adam-bias-correction.png': [960, 1085],
  '/images/notes/hands-on-ml/training-deep-networks/optimizer-comparison.png': [776, 421],
  '/images/notes/hands-on-ml/training-deep-networks/cosine-vs-exponential.png': [960, 355],
  '/images/notes/hands-on-ml/training-deep-networks/cosine-restarts.png': [960, 359],
  '/images/notes/hands-on-ml/training-deep-networks/one-cycle.png': [960, 525],
  '/images/notes/hands-on-ml/training-deep-networks/dropout.png': [882, 593],
  '/images/notes/hands-on-ml/training-deep-networks/mc-dropout.png': [960, 609],
  '/images/notes/hands-on-ml/training-deep-networks/practical-guidelines.png': [990, 480],
  '/images/notes/hands-on-ml/training-deep-networks/dying-relu.png': [2030, 320],
};

export async function renderMarkdown(markdown: string, minimumHeading = 2): Promise<RenderedMarkdown> {
  const headings: Heading[] = [];
  const result = await unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkMath)
    .use(normalizeHeadings, { minimum: minimumHeading })
    .use(remarkRehype)
    .use(rehypeSlug)
    .use(function collectHeadingsAndAssets() {
      return (tree: unknown) => {
        walk(tree as TreeNode, node => {
          if (node.type !== 'element') return;
          const properties = node.properties ?? (node.properties = {});
          if (/^h[2-6]$/.test(node.tagName ?? '')) {
            headings.push({ id: String(properties.id), text: nodeText(node), level: Number(node.tagName?.slice(1)) });
          }
          if (node.tagName === 'img') {
            properties.loading = 'lazy';
            properties.decoding = 'async';
            const dimensions = imageDimensions[String(properties.src)];
            if (dimensions) [properties.width, properties.height] = dimensions;
          }
          if (node.tagName === 'pre') {
            properties.tabIndex = 0;
            properties.role = 'region';
            properties.ariaLabel = 'Code example';
          }
        });
      };
    })
    .use(rehypeKatex)
    .use(rehypeHighlight)
    .use(function wrapTablesAndMath() {
      return (tree: unknown) => {
        const wrap = (node: TreeNode) => {
          if (!node.children) return;
          node.children = node.children.map(child => {
            wrap(child);
            if (child.type === 'element' && Array.isArray(child.properties?.className) &&
                child.properties.className.includes('katex-error')) {
              throw new Error(`Invalid LaTeX: ${nodeText(child)}`);
            }
            if (child.type === 'element' && child.tagName === 'table') {
              return {
                type: 'element', tagName: 'div',
                properties: { className: ['table-scroll'], tabIndex: 0, role: 'region', ariaLabel: 'Scrollable table' },
                children: [child],
              };
            }
            if (child.type === 'element' && Array.isArray(child.properties?.className) &&
                child.properties.className.includes('katex-display')) {
              child.properties.tabIndex = 0;
              child.properties.role = 'group';
              child.properties.ariaLabel = 'Equation';
            }
            return child;
          });
        };
        wrap(tree as TreeNode);
      };
    })
    .use(rehypeStringify)
    .process(markdown);
  const mathError = result.messages.find(message => message.source === 'rehype-katex');
  if (mathError) throw new Error(`Invalid LaTeX: ${String(mathError.cause ?? mathError.message)}`);
  return {
    html: String(result), headings,
    readingMinutes: Math.max(1, Math.ceil(markdown.trim().split(/\s+/).length / 200)),
  };
}

export async function markdownToHtml(markdown: string) {
  return (await renderMarkdown(markdown, 3)).html;
}

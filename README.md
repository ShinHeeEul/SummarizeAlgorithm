# 알고리즘 공부 정리


# 안정 정렬 vs 불안정 정렬


## 안정 정렬 (Stable Sort)
- 안정 정렬은 중복된 값을 입력 순서와 동일하게 정렬하는 정렬 알고리즘
- ex) 삽입 정렬, 병합 정렬, 버블 정렬

## 불안정 정렬 (Unstable Sort)
- 중복된 값이 입력 순서와 동일하지 않게 정렬되는 알고리즘
- ex) 퀵 정렬, 선택 정렬, 계수 정렬


# 정렬 종류

## 버블 정렬 (Bubble Sort)

## 계수 정렬 (Counting Sort)

## 힙 정렬 (Heap Sort)

## 삽입 정렬 (Insertion Sort)

## 합병 정렬 (Merge Sort)

## 퀵 정렬 (Quick Sort)

## 기수 정렬 (Radix Sort)

## 선택 정렬 (Selection Sort)

## 위상 정렬(Topological Sort)
- 선후 관계가 정의된 그래프 구조(순서가 정해진 작업)에서 정렬을 하기 위해 사용
- 끝난 후 진입차수가 0이 아닌 노드가 있다면 순환이 존재한다는 것
- 시간 복잡도 : O(V + E)

1) 그래프의 각 노드들의 진입 차수 테이블 생성 및 진입 차수 계산
2) 진입 차수가 0인 노드를 큐에 넣기
3) 큐에서 노드를 꺼내고 꺼낸 노드 출력
4) 꺼낸 노드와 간선으로 연결된 노드들의 진입 차수--
5) 진입 차수가 0인 노드가 있다면 큐에 넣기
6) 3~5를 큐에 더 이상 아무것도 없을 때까지 반복

- 코드 템플릿

```java
import java.util.LinkedList;

class Topological {
  ArrayList<Integer>[] graph = new ArrayList[N+1];
  Queue<Integer> q = new LinkedList<>();
  int[] edgeCount = new int[9];
  
  public void topological() {
    
    for(int i = 1; i < N; i++) {
        if(edgeCount[i] == 0) {
            q.offer(i);
        }
    }
    
    while(!q.isEmpty()) {
        int nodeNo = q.poll();
        
        List<Integer> list = graph.get(nodeNo);
        
        for(int i = 0; i < list.size(); i++) {
            if(--edgeCount[list.get(i)] == 0) {
                q.offer(list.get(i));
            }
        }
    }
  }
}
```

# DFS vs BFS
- 너무 쉽다 이건


# 이분탐색
- 시간복잡도 : O(logN)

# 유니온-파인드(Union Find)
- 그래프 알고리즘으로 두 노드가 같은 그래프에 속하는지 판별하는 알고리즘.
- 서로소 집합, 상호 베타적 집합, 분리 집합으로도 불림
- union(x,y) 연산과 find(x) 연산으로 이루어져 있다.
- union(x,y) 연산은 x와 y의 그래프를 합친다는 의미이며, find(x)는 x가 어느 그래프에 속하는지 연산한다는 의미이다.
- 일반적으로 O(logN), 최대 O(N), 경로 압축시 O(a(N)). a(N)은 아커만 함수를 의미 : 상수 시간 복잡도로 생각해도 됨

- union(x,y)
  1) x와 y의 부모노드를 찾는다.
  2) 둘이 같은 그래프라면 false를 반환하고
  3) x가 y보다 작다면 arr[y] = x, 반대라면 arr[x] = y

- find(x)
  1) parent[x] == x라면 x를 반환하고
  2) 아니라면 find(parent[x]);
     - 이 과정에서 parent[x] = find(parent[x])를 반환하면 경로 압축을 할 수 있다.

- 코드 템플릿
```java
class UnionFind {
    
    boolean union(int x, int y) {
        x = find(x);
        y = find(y);
        
        if(x == y) {
            return false;
        }
        if(x < y) {
            arr[x] = y;
        } else {
            arr[y] = x;
        }
        
        return true;
    }
    
    int find(int x) {
        if(x == arr[x]) {
            return arr[x];
        }
        return arr[x] = find(arr[x]);
    }
}
```

# 그래프 최단 거리 구하는 알고리즘들
## 밸만-포드 알고리즘
- 그래프 최단 경로 구하는 알고리즘
- 하나의 정점에서 출발하는 최단 거리를 구하며 음수 사이클이 없어야 함.(음수 가중치는 허용)
- 다익스트라에 비해 조금 더 시간복잡도가 증가한 대신 음수 가중치를 처리할 수 있음.
- 시간복잡도 : O(EV)

1) 출발 노드는 0 나머지는 INF로 초기화
2) 간선 m개를 하나씩 살펴보며 dist[v]가 무한대가 아닐 때 dist[v] = min(dist[v], dist[w] + cost(w,v))
3) 1)~2)과정을 모든 노드간 반복

- 음수 사이클 확인하기
  - n개 만큼 반복하는 과정을 한번 더 했을 때 바뀌는 값이 있다면 음수 사이클이 존재하는 것.

- 코드 템플릿
```java
class BellmanFord {
    class Edge {
        int v; //나가는 노드
        int w; //들어오는 노드
        int cost;
        
        public Edge(int v, int w, int cost) {
            this.v = v;
            this.w = w;
            this.cost = cost;
        }
    }
    
    static ArrayList<Edge> graph;
    
    public boolean BellmanFord(int n, int m, int start) {
        int[] dist = new int[n + 1];
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[start] = 0;
        
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                Edge edge = graph.get(j);
                if(dist[edge.v] != Integer.MAX_VALUE) {
                    dist[edge.w] = Math.min(dist[edge.w], dist[edge.v] + edge.cost);
                }
            }
        }

        //음수 사이클 검증
        for(int j = 0; j < m; j++) {
            Edge edge = list.get(j);
            if(dist[edge.v] != Integer.MAX_VALUE && dist[edge.w] > dist[edge.v] + edge.cost) {
                System.out.println("음수");
                return;
            }

        }
    }
}
```
## 플로이드-워셜 알고리즘
- 음수 사이클이 없는 그래프 내의 모든 정점에서 각 모든 정점에 까지의 최단거리를 모두 구할 수 있는 알고리즘.
- DP랑, 인접 행렬 이용
- 모든 노드에서 모든 노드로 가는 최소 비용을 단계적으로 갱신하면서 진행되는 알고리즘
- 시간복잡도 : O(N ^ 3)

- 코드 템플릿
```java
public class FloydWarshall {
    
    public static void floydWarshall() {
        int[][] D = new int[N][N];
        
        //갈 수 없는 경로 확인
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                if(i == j) continue;
                if(D[i][j] == 0) D[i][j] = Integer.MAX_VALUE;
            }
        }
        
        //플로이드 워셜
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                if(i == j) continue; //출발지와 경유지가 같으면 다음 탐색
                
                for(int k = 0; k < N; k++) {
                    if(i == k || j == k) continue; //출발지와 도착지가 같거나 도착지가 경유지면 다음 탐색
                    D[j][k] = Math.min(D[j][i] + D[i][k], D[j][k]); // 경유하거나 직접가거나 더 짧은 경로로 대체
                }
            }
        }
    }
}
```

## 다익스트라 알고리즘
- 음의 가중치가 없는 그래프의 한 노드에서 각 모든 노드까지의 최단거리를 구하는 알고리즘을 말함.
- 기본적으로 그리디 + 다이나믹 프로그래밍 기법
- 시간복잡도 : O(ElogV)

1) 아직 방문하지 않은 정점 중 출발지로부터 가장 거리가 짧은 정점을 방문
2) 해당 정점을 거쳐 갈 수 있는 정점의 거리가 이전 기록한 값보다 적으면 갱신

- 코드 템플릿

```java
import java.util.PriorityQueue;

class Dijkstra {
    class Node implements Comparable<Node> {
        int index;
        int cost;

        public Node(int index, int cost) {
            this.index = index;
            this.cost = cost;
        }

        @Override
        public int compareTo(Node o) {
            return Integer.compare(this.cost, o.cost);
        }
    }

    static ArrayList<Node>[] graph;

    public void Dijkstra(int n, int start) {
        boolean[] check = new boolean[n + 1];
        int[] dist = new int[n + 1];
        final int INF = Integer.MAX_VALUE;
        Arrays.fill(dist, INF);
        dist[start] = 0;

        PriorityQueue<Node> pq = new PriorityQueue<>();
        pq.offer(new Node(start, 0));
        
        while(!pq.isEmpty()) {
            int nowVertex = pq.poll().index;
            
            if(check[nowVertex]) continue;
            check[nowVertex] = true;
            
            for(Node next : graph[nowVertex]) {
                if(dist[next.index] > dist[nowVertex] + next.cost) {
                    dist[next.index] = Math.min(dist[next.index], dist[nowVertex] + next.cost);
                    pq.offer(new Node(next.index, dist[next.index]));
                }
            }
        }
    }
}
```


# LCS(Longest Common Subsequence) : 최장 부분 수열 알고리즘
- dp를 응용한 최장 부분 수열 알고리즘
- 시간복잡도 : O(nm)

1) arr을 2차원 배열(arr)로 선언
2) 문자를 하나씩 비교
   1) 일치하는 문자가 나오면 arr[i][j] = arr[i-1][j-1] + 1
   2) 일치하지 않는 문자가 나온다면 arr[i][j] = max(arr[i-1][j], arr[i][j-1])
3) 배열이 끝날때까지 반복

- LCS 값을 구하는 경우 이를 역추적
1) arr[i-1][j] or arr[i][j-1] 중 같은 값이 있다면 거기로 이동
2) 없다면 arr[i-1][j-1]로 이동하면서 arr[i][j] 값을 answer 문자열 앞에 더하기
3) 반복이 끝나면 answer 출력

- 코드 템플릿
```java
public class LCS {
    static int LCS(int N) {
        String s1;
        String s2;
        int[][] dp = new int[N][N];
        
        for(int i = 1; i < N; i++) {
            for(int j = 1; j < N; j++) {
                if(s1.charAt(i-1) == s2.charAt(j-1)) {
                    dp[i][j] = dp[i-1][j-1] + 1;
                }
                else {
                    dp[i][j] = Math.max(dp[i][j-1], dp[i-1][j]);
                }
            }
        }
    }
}

```

# LIS(Longest Increasing Subsequence) : 최장 증가 부분 수열
- 어떠한 수열에서 오름차순으로 증가하는 가장 긴 부분수열을 찾는 것

## dp를 이용한 방법
   - 정확한 LIS를 구할 때 사용
   - 시간 복잡도 : O(N ^ 2)
   1) k의 앞 순서에 있는 모든 원소들 중 값이 k보다 작은 원소에 대해, 각각의 원소에서 끝나는 최장 증가 수열의 길이를 알고 있으면 구할 수 있음.
   2) dp[i] = i번째 인덱스에서 끝나는 최장 증가 수열의 길이

## 이분 탐색을 이용한 방법
   - 정확한 LIS가 아닌 길이만 구할 때 사용
   - 시간 복잡도 : O(NlogN)
   1) 배열을 하나 더 두고 원래 수열에서의 각 원소에 대해 LIS 배열 내에서의 위치를 찾음.
   2) 기존 배열을 탐색 
   2) 현재 수가 배열의 마지막 값보다 크다면 LIS 내 배열에 추가
   3) 현재 수가 배열의 마지막 값보다 작다면 이분 탐색을 통해 들어갈 위치를 찾고 교체.
      1) 이분 탐색 시, start = 0, end = 현재 위치이며 start < end동안 반복 -> end 값으로 교체

- 코드 템플릿
```java
// 이분 탐색으로 길이만 구함
public class LIS {
    static void LIS(int N) {
        int[] arr = new int[N];
        int[] lis = new int[N];
        int current = 0;
        for(int i = 0; i < N; i++) {
            int target = arr[0];
            if(i == 0) {
                lis[0] = target;
                continue;
            } 
            if(target > lis[current]) {
                lis[++current] = target;
            } else {
                int start = 0;
                int end = current;
                while(start < end) {
                    int mid = (start + end) >>> 1;
                    if(lis[mid] < target) {
                        start = mid + 1;
                    } else {
                        end = mid;
                    }
                }
                lis[end] = target;
            }
        }
    }
}
```
# 투 포인터

# MST(Minimum Spanning Tree)
- Spanning Tree란?
  - 그래프 내의 모든 정점을 포함하는 트리
    - 최소 연결 = 간선의 수가 가정 적다
    - n개의 정점을 가지는 그래프의 최소 간선의 수는 (n-1)개이고, (n-1)개의 간선으로 연결되어 있으면 필연적으로 트리 형태가 되고 이것이 Spanning Tree
- DFS, BFS을 이용하여 그래프에서 신장 트리를 찾을 수 있음.
- 모든 정점들이 연결되어야 하고, 사이클을 포함해서는 안됨.

- MST(Minimum Spanning Tree)란?
  - Spanning Tree 중에서 사용된 간선들의 가중치 합이 최소인 트리
  - 네트워크에 있는 모든 정점들을 가장 적은 수의 간선과 비용으로 연결하는 것
- 특징
  - 간선의 가중치의 합이 최소여야 함
  - n개의 정점을 가지는 그래프에 대해 반드시 (n-1)개의 간선만을 사용해야 함
  - 사이클이 포함되어서는 안됨
- 구현 방법
  1) Kruskal MST 알고리즘
    - 그리디 알고리즘을 이용하여 네트워크의 모든 정점을 최소 비용으로 연결하는 최적 해답을 구하는 법
    - Union-Find 알고리즘을 활용하여 구현
    - 시간 복잡도 : O(ElogE)
  
    - 동작 방식
      1) 그래프의 간선들을 가중치 기준 오름차순으로 정렬
      2) 정렬된 간선 리스트를 순서대로 선택, 간선의 정점들을 연결
      3) 정점을 연결하는 것은 Union-Find의 Union으로 구현
      4) 간선의 두 정점 a,b가 이미 연결되어 있다면 스킵
      5) 위 과정을 반복하여 최소 비용의 간선들만 이용하여 모든 정점이 연결됨
  2) Prim MST 알고리즘
  - 시작 정점에서부터 출발하여 신장트리 집합을 단계적으로 확장해나가는 방법
  - 우선순위 큐를 활용하여 구현
  - 시간복잡도 : O(ElogV)

  - 동작 방식
    1) 임의의 정점을 시작점으로 선택
    2) 갈 수 있는 정점 중 가장 가중치가 작은 정점 연결
    3) 새로운 정점 중 가중치가 가장 작은 정점으로 연결

  - 코드 템플릿

```java
  import java.util.PriorityQueue;

class Prim {
    static class Edge implements Comparable<Edge> {
        Edge(int w, int cost) {
            this.w = w;
            this.cost = cost;
        }

        @Override
        public int compareTo(Edge o) {
            return this.cost - o.cost;
        }
    }
    
    static List<Edge>[] graph;

    public static void prim(int start, int n) {
        boolean[] visit = new boolean[n + 1];

        PriorityQueue<Edge> pq = new PriorityQueue<>();
        pq.offer(new Edge(start, 0));
        
        int total = 0;
        while(!pq.isEmpty()) {
            Edge edge = pq.poll();
            
            int v = edge.w;
            int cost = edge.cost;
            
            if(visit[v]) continue;
            
            visit[v] = true;
            total += cost;
            
            for(Edge e : graph[v]) {
                if(!visit[e.w]) {
                    pq.add(e);
                }
            }
        }
    }
}
  ```
# Bit Masking

# DP(Dynamic Programming)
- DP란?
  - 하나의 큰 문제를 작은 문제로 나누어 해결하는 기법
- 구현 방법
  1) 먼저 DP로 풀 수 있는 문제인지 확인.
  2) 각 소문제들을 나누는 변수를 정함.
  3) 각 변수들간의 관계식, 즉 점화식을 찾기.
  - Top-Down
    - 주로 재귀함수를 사용하며 dp[n]값을 찾기 위한 재귀 함수의 호출이 dp[0]까지 내려간 다음 결과들이 재귀 함수에 맞물리며 재활용되는 방식.
    ```
    dp = Math.min(재귀함수 경로1() + value, 재귀함수 경로2() + value);
    ```
  - Bottom-Up
    - 초기조건을 기반으로 차곡차곡 데이터를 쌓아가 큰 문제의 결과를 도출하는 과정
- 정형화된 코드 템플릿은 없는 듯하다.

# 냅색(Knapsack)
-  Knapsack Problem, 배낭문제는 다이나믹 프로그래밍에서 매우 유명한 문제
- 냅색이란?
  - 어떤 배낭이 있고 그 배낭안에 넣을 수 있는 최대 무게가 K라고 하자. 배낭에 넣을 수 있는 N개의 물건이 각기 다른 가치 V를 가지고 있고 각 물건마다 다른 무게 W를 가지고 있을 때, 배낭이 최대한 가치가 높은 물건들을 담을 수 있는 조합을 찾는 문제이다.
  - 해당 문제는 물건을 쪼갤 수 있는 Fraction Knapsack Problem과 물건을 쪼갤 수 없는 0-1 knapSack Problem으로 나뉜다.
- [0-1 Knapsack Problem](https://howudong.tistory.com/106)
- 물건 K의 무게 > 배낭 W 무게
  - dp [K][W] = dp [K-1][W]
- 물건 K의 무게 <= 배낭 W 무게
  - dp [K][W] = max(dp [K-1][W], K가치 + dp [K-1][W-K무게])
